# -*- coding: utf-8 -*-
import os
import threading
from typing import Any, Dict, Generator, List, Optional, Tuple
import time
from pathlib import Path
from PIL import Image

import torch

from src.utils.input_utils import build_image
from src.utils.model_utils import build_emu3p5
from src.utils.generation_utils import generate, multimodal_decode


class ModelRuntime:
    """
    统一管理模型实例、上下文与流式生成；支持：
    ✅ 初始化/热重载（不重启 App）
    ✅ 配置切换时，如 model_path/tokenizer_path/vq_path 不同 → 自动 reload 模型
    ✅ 清空历史/选择性裁剪历史（显存保护）
    ✅ streaming 文本分块输出 + 图片 ready 即推送
    ✅ Stop 中断生成
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
        self._main_cfg_path: Optional[str] = None

        # 显存管理 / 历史裁剪策略
        self.context_limit_tokens: int = 8192
        self.history_keep_last_steps: int = 6
        self.max_new_tokens_cap: int = 2048
        self.enable_cuda_empty_cache: bool = True

        # 运行时上下文 / 历史 prompt trace
        self.history: List = []

    # ---------- 动态载入 config ----------
    def _load_cfg_module(self, cfg_path: str):
        import importlib.util
        from pathlib import Path

        cfg_path = os.path.abspath(cfg_path)
        spec = importlib.util.spec_from_file_location(Path(cfg_path).stem, cfg_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)  # type: ignore
        return module

    # ---------- 模型初始化/重载 ----------
    def initialize(self, cfg_path: str, save_dir: str,
                   device_str: Optional[str] = None,
                   force_reload: bool = False) -> str:

        if self.model is not None and not force_reload:
            return "✅ 模型已就绪（复用现有实例）"

        # 卸载旧模型
        if self.model is not None:
            try:
                self.model = None
                self.tokenizer = None
                self.vq_model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass

        cfg = self._load_cfg_module(cfg_path)

        if device_str:
            device = torch.device(device_str)
        else:
            device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

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

        self._apply_memory_cfg_overrides(cfg)

        self.cfg_module = cfg
        self._main_cfg_path = cfg_path
        self._device = device
        self._save_dir = save_dir
        self.history = []

        return f"✅ 模型已加载到 {device}，输出目录：{save_dir}"

    # ---------- 关键增强功能：判断是否需要重载模型 ----------
    def update_sampling_config(self, mode: str) -> None:
        """
        ✅ 如果 config 中模型权重（model_path / tokenizer_path / vq_path）与当前不一致，
           自动初始化 model（热重载）
        ⚡ 如果一致，仅更新 sampling 参数，不 reload 模型
        """
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
            print(f"[config change] reload model due to different weights. ({cfg_file})")
            self.initialize(
                cfg_path=cfg_file,
                save_dir=self._save_dir or "./outputs",
                device_str=str(self._device) if self._device else None,
                force_reload=True
            )
        else:
            for key in self._sampling_keys:
                if hasattr(new_cfg, key):
                    setattr(self.cfg_module, key, getattr(new_cfg, key))

            print(f"[sampling update] config changed ({cfg_file}), model reused")

    # ---------- 运行控制 ----------
    def clear_history(self) -> None:
        self.history = []

    def request_stop(self) -> None:
        self._stop_event.set()

    def reset_stop(self) -> None:
        self._stop_event.clear()

    # ---------- Prompt 编码 ----------
    def encode_and_set_prompt(self, sample: Dict[str, Any]) -> None:
        self.clear_history()
        seq = self.encode_prompt(sample)
        self.history.append(seq)

        user_dir = os.path.join(self._save_dir, "sessions")
        os.makedirs(user_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        idx = len(os.listdir(user_dir))
        session_dir = os.path.join(user_dir, f"session_{timestamp}_{idx:04d}")
        os.makedirs(session_dir, exist_ok=True)
        self._current_session_dir = session_dir

        text_prompt = sample.get("text_prompt", "")
        with open(os.path.join(session_dir, "prompt.txt"), "w", encoding="utf-8") as f:
            f.write(text_prompt)

        images = sample.get("images", [])
        for i, img_path in enumerate(images):
            try:
                img = Image.open(img_path)
                img.save(os.path.join(session_dir, f"image_{i:02d}.png"))
            except:
                pass

    def encode_prompt(self, sample: Dict[str, Any]):
        text_prompt = sample.get("text", "")
        images = sample.get("images", [])

        cfg = self.cfg_module
        unc_prompt, template = cfg.build_unc_and_template(cfg.task_type, with_image=bool(images))

        if images:
            image_str = ""
            for img_path in images:
                img = Image.open(img_path).convert("RGB")
                image_str += build_image(img, cfg, self.tokenizer, self.vq_model)
            prompt = template.format(question=text_prompt).replace("<|IMAGE|>", image_str)
            unc_prompt = unc_prompt.replace("<|IMAGE|>", image_str)
        else:
            prompt = template.format(question=text_prompt)

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        unconditional_ids = self.tokenizer.encode(unc_prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)

        if input_ids[0, 0] != cfg.special_tokens["BOS"]:
            BOS = torch.tensor([[cfg.special_tokens["BOS"]]], device=input_ids.device, dtype=input_ids.dtype)
            input_ids = torch.cat([BOS, input_ids], dim=1)

        return input_ids, unconditional_ids

    # ---------- 显存管理 ----------
    def _apply_memory_cfg_overrides(self, cfg_module: Any) -> None:
        try:
            if hasattr(cfg_module, "context_limit_tokens"):
                self.context_limit_tokens = int(cfg_module.context_limit_tokens)
            if hasattr(cfg_module, "history_keep_last_steps"):
                self.history_keep_last_steps = int(cfg_module.history_keep_last_steps)
            if hasattr(cfg_module, "max_new_tokens_cap"):
                self.max_new_tokens_cap = int(cfg_module.max_new_tokens_cap)
            if hasattr(cfg_module, "enable_cuda_empty_cache"):
                self.enable_cuda_empty_cache = bool(cfg_module.enable_cuda_empty_cache)
        except:
            pass

    def _maybe_trim_history(self) -> None:
        if not self.history:
            return

        if len(self.history) > self.history_keep_last_steps:
            self.history = self.history[-self.history_keep_last_steps:]

        total_tokens = sum(len(seq) for seq in self.history)
        while total_tokens > self.context_limit_tokens and len(self.history) > 1:
            self.history.pop(0)
            total_tokens = sum(len(seq) for seq in self.history)

    # ---------- Streaming 输出 ----------
    def stream_events(self, max_rounds: int = 32, text_chunk_tokens: int = 64) -> Generator[Dict[str, Any], None, None]:
        assert self.model and self.tokenizer and self.vq_model and self.cfg_module

        if not self.history:
            raise RuntimeError("No prompt set. 请先调用 encode_and_set_prompt。")
        input_ids, _ = self.history[-1]

        unc_prompt = getattr(self.cfg_module, "unc_prompt", "")
        unconditional_ids = self.tokenizer.encode(unc_prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)

        full_unc_ids = self.tokenizer.encode(self.cfg_module.img_unc_prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device) \
                        if hasattr(self.cfg_module, "img_unc_prompt") else None

        text_accum = ""
        session_dir = getattr(self, "_current_session_dir", self._save_dir)
        image_chunk_idx = 0

        for result_tokens in generate(self.cfg_module, self.model, self.tokenizer, input_ids, unconditional_ids, full_unc_ids, True):
            if self._stop_event.is_set():
                yield {"type": "text", "text": "[Stopped by user]"}
                break

            try:
                result = self.tokenizer.decode(result_tokens, skip_special_tokens=False)
                mm_out = multimodal_decode(result, self.tokenizer, self.vq_model)

                for item in mm_out:
                    if item[0] == "text":
                        text_chunk = item[1][:text_chunk_tokens]
                        yield {"type": "text", "text": text_chunk}

                    elif item[0] == "image":
                        img = item[1]
                        img_path = os.path.join(session_dir, f"gen_image_{image_chunk_idx:03d}.png")
                        img.save(img_path)
                        image_chunk_idx += 1
                        yield {"type": "image", "paths": [img_path]}

            except Exception as e:
                yield {"type": "text", "text": f"[ERROR] {e}"}
                break

    @property
    def save_dir(self) -> str:
        assert self._save_dir is not None
        return self._save_dir