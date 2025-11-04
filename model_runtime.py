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
    ✅ 初始化 / 热重载（仅模型权重变化才重建模型）
    ✅ config 变化时保留运行时动态覆盖的 sampling 参数（top_p/max_new_tokens等）
    ✅ Streaming (逐token、逐图片输出)
    ✅ Stop 优雅中断（停止在 next chunk 返回后）
    """

    _singleton: Optional["ModelRuntime"] = None

    # 支持 UI 动态覆盖的参数列表
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

        # ⭐ 存储外部设置的 sampling 参数（不会因为 config reload 丢掉）
        self.runtime_cfg_overrides: Dict[str, Any] = {}

        # memory / history
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

    # ------------------- 模型初始化 / 热重载 -------------------
    def initialize(self, cfg_path: str, save_dir: str,
                   device_str: Optional[str] = None,
                   force_reload: bool = False) -> str:

        cfg = self._load_cfg_module(cfg_path)

        # ★ 重新 load config 时，把 runtime 保存的 override 参数都 apply 回 config ★
        for k, v in self.runtime_cfg_overrides.items():
            setattr(cfg, k, v)

        # ----------- 仅当权重变化时才重载模型 ----------- #
        def weights(c):
            return (c.model_path, c.tokenizer_path, c.vq_path)

        model_changed = (
            self.model is None or force_reload or
            (self.cfg_module is not None and weights(self.cfg_module) != weights(cfg))
        )

        if model_changed:
            # clean gpu memory
            if self.model is not None:
                self.model = None
                self.tokenizer = None
                self.vq_model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

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

            # auto build special_token_ids  (special_tokens->token_id)
            cfg.special_token_ids = {
                k: self.tokenizer.convert_tokens_to_ids(v)
                for k, v in cfg.special_tokens.items()
            }

            self._device = device

        # ----------- 每次 config 变化也要更新 sampling 参数 ----------- #
        self._apply_memory_cfg_overrides(cfg)

        save_dir = os.path.abspath(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        self.cfg_module = cfg
        self._save_dir = save_dir
        self.history = []

        return f"✅ 模型已加载/更新 正在使用 device={self._device}"

    # ------------------- UI 根据 mode 切换 config -------------------
    def update_sampling_config(self, mode: str) -> None:
        config_map = {
            "howto": "configs/example_config_visual_guidance.py",
            "story": "configs/example_config_visual_narrative.py",
            "t2i": "configs/example_config_t2i.py",
            "x2i": "configs/example_config_x2i.py",
            "default": "configs/config.py",
        }
        cfg_file = config_map.get(mode, "configs/config.py")

        print(f"[mode-change] → {mode} ({cfg_file})")
        self.initialize(cfg_file, save_dir=self._save_dir, device_str=str(self._device))

    # ------------------- 外部动态覆盖 sampling 参数 -------------------
    def update_runtime_cfg(self, **kwargs):
        """
        供 UI 动态设置 max_new_tokens/top_p 等
        """
        for k, v in kwargs.items():
            self.runtime_cfg_overrides[k] = v
            setattr(self.cfg_module, k, v)

    # ------------------- 控制 API -------------------
    def clear_history(self): self.history = []
    def request_stop(self): self._stop_event.set()
    def reset_stop(self): self._stop_event.clear()

    # ------------------- prompt 编码 -------------------
    def encode_and_set_prompt(self, sample: Dict[str, Any]) -> None:
        self.clear_history()
        input_ids, unconditional_ids = self.encode_prompt(sample)
        self.history.append((input_ids, unconditional_ids))

        # 保存 session
        user_dir = os.path.join(self._save_dir, "sessions")
        os.makedirs(user_dir, exist_ok=True)
        session_dir = os.path.join(user_dir, f"session_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(session_dir, exist_ok=True)
        self._current_session_dir = session_dir

    def encode_prompt(self, sample: Dict[str, Any]):
        cfg = self.cfg_module
        text_prompt = sample.get("text", "")
        images = sample.get("images", [])

        unc_prompt, template = cfg.build_unc_and_template(cfg.task_type, with_image=bool(images))

        # encode images
        if images:
            img_str = "".join(
                build_image(Image.open(p).convert("RGB"), cfg, self.tokenizer, self.vq_model)
                for p in images
            )
            prompt = template.format(question=text_prompt).replace("<|IMAGE|>", img_str)
            unc_prompt = unc_prompt.replace("<|IMAGE|>", img_str)
        else:
            prompt = template.format(question=text_prompt)

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(self._device)
        unconditional_ids = self.tokenizer.encode(unc_prompt, return_tensors="pt", add_special_tokens=False).to(self._device)

        return input_ids, unconditional_ids

    # ------------------- Streaming 输出（Stop 优雅中断） -------------------
    def stream_events(self, text_chunk_tokens: int = 64) -> Generator[Dict[str, Any], None, None]:
        assert self.history, "请先调用 encode_and_set_prompt()"
        input_ids, unconditional_ids = self.history[-1]
        session_dir = getattr(self, "_current_session_dir", self._save_dir)
        img_idx = 0

        for tokens in generate(self.cfg_module, self.model, self.tokenizer,
                               input_ids, unconditional_ids, None, True):

            result_str = self.tokenizer.decode(tokens, skip_special_tokens=False)
            parsed = multimodal_decode(result_str, self.tokenizer, self.vq_model)

            for typ, val in parsed:
                if typ == "text":
                    yield {"type": "text", "text": val[:text_chunk_tokens]}

                elif typ == "image":
                    img_path = os.path.join(session_dir, f"gen_{img_idx:03d}.png")
                    val.save(img_path)
                    img_idx += 1
                    yield {"type": "image", "paths": [img_path]}

            if self._stop_event.is_set():
                self.reset_stop()
                yield {"type": "text", "text": "🛑 用户请求停止，已中断。"}
                break

    @property
    def save_dir(self) -> str:
        assert self._save_dir is not None
        return self._save_dir
