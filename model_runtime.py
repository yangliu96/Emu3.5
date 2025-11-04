# -*- coding: utf-8 -*-
import os
import threading
import time
from typing import Any, Dict, Generator, List, Optional
from pathlib import Path
from PIL import Image

import torch

from src.utils.input_utils import build_image
from src.utils.model_utils import build_emu3p5
from src.utils.generation_utils import generate, multimodal_decode


class ModelRuntime:
    """
    统一管理模型实例、上下文与流式生成；支持：
    ✅ 初始化 / 热重载
    ✅ 配置变化（model_path/tokenizer_path/vq_path）自动 reload
    ✅ Streaming (逐 token、逐图片输出)
    ✅ Stop 优雅中断（输出当前 chunk 后停止）
    """

    _singleton: Optional["ModelRuntime"] = None

    _sampling_keys = [
        "top_p", "top_k", "temperature", "num_beams", "max_new_tokens",
        "min_new_tokens", "repetition_penalty", "do_sample", "steps"
    ]

    @classmethod
    def instance(cls) -> "ModelRuntime":
        if cls._singleton is None:
            cls._singleton = ModelRuntime()
        return cls._singleton

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.vq_model = None

        self.cfg_module: Optional[Any] = None
        self._device: Optional[torch.device] = None
        self._save_dir: Optional[str] = None
        self._stop_event = threading.Event()

        # memory 优化
        self.context_limit_tokens: int = 8192
        self.history_keep_last_steps: int = 6
        self.max_new_tokens_cap: int = 2048
        self.enable_cuda_empty_cache: bool = True

        # history 保存 (input_ids, unconditional_ids)
        self.history: List = []

    # ------------------- config 动态加载 -------------------
    def _load_cfg_module(self, cfg_path: str):
        import importlib.util
        cfg_path = os.path.abspath(cfg_path)
        spec = importlib.util.spec_from_file_location(Path(cfg_path).stem, cfg_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module

    # ------------------- 模型初始化 / 重载 -------------------
    def initialize(self, cfg_path: str, save_dir: str,
                   device_str: Optional[str] = None,
                   force_reload: bool = False) -> str:

        if self.model is not None and not force_reload:
            return "✅ 模型已就绪（复用实例）"

        # clean gpu
        if self.model is not None:
            self.model = None
            self.tokenizer = None
            self.vq_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        cfg = self._load_cfg_module(cfg_path)

        device = torch.device(device_str) if device_str else (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )

        # build emu3.5
        self.model, self.tokenizer, self.vq_model = build_emu3p5(
            cfg.model_path,
            cfg.tokenizer_path,
            cfg.vq_path,
            vq_type=getattr(cfg, "vq_type", None),
            model_device=getattr(cfg, "hf_device", device),
            vq_device=getattr(cfg, "vq_device", device),
            **getattr(cfg, "diffusion_decoder_kwargs", {}),
        )

        save_dir = os.path.abspath(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        self.cfg_module = cfg
        self._device = device
        self._save_dir = save_dir
        self.history = []

        return f"✅ 模型已加载到 {device} 输出目录：{save_dir}"

    # ------------------- sampling config / hot reload -------------------
    def update_sampling_config(self, mode: str) -> None:
        config_map = {
            "howto": "configs/example_config_visual_guidance.py",
            "story": "configs/example_config_visual_narrative.py",
            "t2i": "configs/example_config_t2i.py",
            "x2i": "configs/example_config_x2i.py",
            "default": "configs/config.py",
        }
        cfg_file = config_map.get(mode, "configs/config.py")
        new_cfg = self._load_cfg_module(cfg_file)

        def _cfg_weights(cfg):
            return (cfg.model_path, cfg.tokenizer_path, cfg.vq_path)

        need_reload = (
            self.cfg_module is None or
            _cfg_weights(self.cfg_module) != _cfg_weights(new_cfg)
        )

        if need_reload:
            print(f"[reload] different model detected -> reload ({cfg_file})")
            self.initialize(cfg_path=cfg_file,
                            save_dir=self._save_dir or "./outputs",
                            device_str=str(self._device) if self._device else None,
                            force_reload=True)
        else:
            for key in self._sampling_keys:
                if hasattr(new_cfg, key):
                    setattr(self.cfg_module, key, getattr(new_cfg, key))
            print(f"[reuse model] sampling config updated ({cfg_file})")

    # ------------------- 控制 API -------------------
    def clear_history(self): self.history = []
    def request_stop(self): self._stop_event.set()
    def reset_stop(self): self._stop_event.clear()

    # ------------------- prompt 编码 -------------------
    def encode_and_set_prompt(self, sample: Dict[str, Any]) -> None:
        self.clear_history()
        input_ids, unconditional_ids = self.encode_prompt(sample)
        self.history.append((input_ids, unconditional_ids))

        # session 保存
        user_dir = os.path.join(self._save_dir, "sessions")
        os.makedirs(user_dir, exist_ok=True)
        session = os.path.join(
            user_dir,
            f"session_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(session, exist_ok=True)
        self._current_session_dir = session

    def encode_prompt(self, sample: Dict[str, Any]):
        cfg = self.cfg_module
        text_prompt = sample.get("text", "")
        images = sample.get("images", [])

        unc_prompt, template = cfg.build_unc_and_template(cfg.task_type, with_image=bool(images))

        if images:
            image_str = "".join(
                build_image(Image.open(p).convert("RGB"), cfg, self.tokenizer, self.vq_model)
                for p in images
            )
            prompt = template.format(question=text_prompt).replace("<|IMAGE|>", image_str)
            unc_prompt = unc_prompt.replace("<|IMAGE|>", image_str)
        else:
            prompt = template.format(question=text_prompt)

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        unconditional_ids = self.tokenizer.encode(unc_prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)

        return input_ids, unconditional_ids

    # ------------------- Streaming 输出（Stop 优雅中断） -------------------
    def stream_events(self, max_rounds: int = 32, text_chunk_tokens: int = 64) -> Generator[Dict[str, Any], None, None]:
        assert self.model and self.tokenizer and self.vq_model and self.cfg_module

        input_ids, unconditional_ids = self.history[-1]
        session_dir = getattr(self, "_current_session_dir", self._save_dir)
        image_chunk_idx = 0

        for result_tokens in generate(self.cfg_module, self.model, self.tokenizer,
                                      input_ids, unconditional_ids, None, True):

            # decode one chunk from model
            try:
                result = self.tokenizer.decode(result_tokens, skip_special_tokens=False)
                outs = multimodal_decode(result, self.tokenizer, self.vq_model)

                for item in outs:
                    if item[0] == "text":
                        yield {"type": "text", "text": item[1][:text_chunk_tokens]}

                    elif item[0] == "image":
                        img = item[1]
                        img_path = os.path.join(session_dir, f"gen_{image_chunk_idx:03d}.png")
                        img.save(img_path)
                        image_chunk_idx += 1
                        yield {"type": "image", "paths": [img_path]}

            except Exception as e:
                yield {"type": "text", "text": f"[ERROR] {e}"}
                break

            # ✅ Stop: 等当前 chunk 输出完再停止
            if self._stop_event.is_set():
                yield {"type": "text", "text": "🛑 已停止生成"}
                self.reset_stop()
                break

    @property
    def save_dir(self) -> str:
        assert self._save_dir is not None
        return self._save_dir