# -*- coding: utf-8 -*-
"""
ModelRuntime: 控制 Emu3.5 推理生命周期：
✅ 启动时预加载模型
✅ 切 config 仅更新生成参数，不 reload 模型
✅ special_token_ids 等外部注入参数不会丢失
✅ Streaming 输出文本 / 图片 + Stop 优雅中断
✅ 保存用户输入与推理结果
"""

import os
import threading
import time
from typing import Any, Dict, Generator, List, Optional
from pathlib import Path
from PIL import Image
import torch

from src.utils.input_utils import build_image
from src.utils.model_utils import build_emu3p5
from src.utils.generation_utils import generate, multimodal_decode   # ✅ 使用修正后的 generate()


class ModelRuntime:
    _singleton: Optional["ModelRuntime"] = None

    _sampling_keys = [
        "top_p", "top_k", "temperature", "num_beams", "max_new_tokens",
        "min_new_tokens", "repetition_penalty", "do_sample"
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
        self.runtime_persist_cfg: Dict = {}

        self._device: Optional[torch.device] = None
        self._save_dir: Optional[str] = None
        self._stop_event = threading.Event()

        self.history: List = []  # [(input_ids, unconditional_ids)]

    # ---------------- config 动态加载 -----------------
    def _load_cfg_module(self, cfg_path: str):
        import importlib.util
        cfg_path = os.path.abspath(cfg_path)
        spec = importlib.util.spec_from_file_location(Path(cfg_path).stem, cfg_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    # ---------------- 模型初始化（启动时调用） -----------------
    def initialize(self, cfg_path: str, save_dir: str,
                   device_str: Optional[str] = None) -> str:

        if self.model is not None:
            return "✅ 模型已就绪（预加载）"

        cfg = self._load_cfg_module(cfg_path)

        device = torch.device(device_str) if device_str else (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.model, self.tokenizer, self.vq_model = build_emu3p5(
            cfg.model_path,
            cfg.tokenizer_path,
            cfg.vq_path,
            vq_type=getattr(cfg, "vq_type", "ibq"),
            model_device=getattr(cfg, "hf_device", device),
            vq_device=getattr(cfg, "vq_device", device),
            **getattr(cfg, "diffusion_decoder_kwargs", {}),
        )

        cfg.special_token_ids = {
            k: self.tokenizer.convert_tokens_to_ids(v)
            for k, v in cfg.special_tokens.items()
        }

        self.runtime_persist_cfg = {
            "special_token_ids": cfg.special_token_ids
        }

        os.makedirs(save_dir, exist_ok=True)

        self.cfg_module = cfg
        self._device = device
        self._save_dir = save_dir

        return f"✅ 模型已加载到 {device}, 输出目录: {save_dir}"

    # ---------------- 切 config 不 reload 模型 -----------------
    def update_sampling_config(self, mode: str):
        config_map = {
            "howto": "configs/example_config_visual_guidance.py",
            "story": "configs/example_config_visual_narrative.py",
            "t2i": "configs/example_config_t2i.py",
            "x2i": "configs/example_config_x2i.py",
            "default": "configs/config.py",
        }

        cfg_file = config_map.get(mode, "configs/config.py")
        new_cfg = self._load_cfg_module(cfg_file)

        for key in self._sampling_keys:
            if hasattr(new_cfg, key):
                setattr(self.cfg_module, key, getattr(new_cfg, key))

        for k, v in self.runtime_persist_cfg.items():
            setattr(self.cfg_module, k, v)

        print(f"[sampling updated] mode={mode}, model reused ✅")


    # ---------------- control 状态 -----------------
    def request_stop(self): self._stop_event.set()
    def reset_stop(self): self._stop_event.clear()

    # ---------------- prompt encode & save 用户输入 -----------------
    def encode_and_set_prompt(self, sample: Dict[str, Any]):
        """保存用户输入 + 建立 session 目录"""

        input_ids, unconditional_ids = self.encode_prompt(sample)
        self.history = [(input_ids, unconditional_ids)]

        session_dir = os.path.join(self._save_dir, f"session_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(session_dir, exist_ok=True)
        self._current_session_dir = session_dir

        # ✅ 保存用户 text 输入
        user_text = sample.get("text", "")
        with open(os.path.join(session_dir, "task.txt"), "w", encoding="utf-8") as f:
            f.write(user_text)

        # ✅ 保存用户 image 输入
        for idx, p in enumerate(sample.get("images", [])):
            try:
                Image.open(p).save(os.path.join(session_dir, f"task_image_{idx}.png"))
            except:
                pass

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

        return (
            self.tokenizer.encode(prompt, return_tensors="pt").to(self._device),
            self.tokenizer.encode(unc_prompt, return_tensors="pt").to(self._device)
        )

    # ---------------- Streaming：保存模型输出 text & image -----------------
    def stream_events(self, text_chunk_tokens: int = 64) -> Generator[Dict[str, Any], None, None]:
        """
        适配新版 generate():
            streaming=False → return ndarray
            streaming=True  → yield {"type": "..."}
        """
        input_ids, unconditional_ids = self.history[-1]
        session_dir = getattr(self, "_current_session_dir", self._save_dir)

        img_idx, text_idx = 0, 0

        # ✅ generate() streaming=True 时 yield event dict
        for ev in generate(self.cfg_module, self.model, self.tokenizer,
                           input_ids, unconditional_ids, None, True):

            if self._stop_event.is_set():
                yield {"type": "text", "text": "🛑 已停止生成"}
                self.reset_stop()
                break

            # ---------------- Streaming 文本事件 ----------------
            if ev["type"] == "text":
                txt = ev["text"][:text_chunk_tokens]
                yield {"type": "text", "text": txt}

                with open(os.path.join(session_dir, f"gen_text_{text_idx}.txt"),
                            "w", encoding="utf-8") as f:
                    f.write(txt)
                text_idx += 1

            # ---------------- Streaming 图片事件 ----------------
            elif ev["type"] == "image":
                img_path = os.path.join(session_dir, f"gen_img_{img_idx}.png")
                ev["image"].save(img_path)
                img_idx += 1

                yield {"type": "image", "paths": [img_path]}

            # ---------------- 最终 token ids （保留原逻辑） ----------------
            elif ev["type"] == "final_ids":
                pass  # 不做 UI 显示，仅保留事件

    @property
    def save_dir(self): 
        return self._save_dir