import os
import threading
from typing import Any, Dict, Generator, List, Optional, Tuple
import time
from PIL import Image

import torch

from src.utils.model_utils import build_emu3p5
from src.utils.generation_utils import generate, multimodal_decode


class ModelRuntime:
    """
    统一管理模型实例、上下文与流式生成；支持：
    - 初始化/热重载（不重启 App）
    - 清空历史/选择性裁剪历史（显存保护）
    - 生成中断（Stop）
    - 分步产出：文本流式分块 + 图像就绪即上传
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
        self._main_cfg_module: Optional[Any] = None

        # 显存/上下文保护参数（可由 cfg 覆盖）
        self.context_limit_tokens: int = 8192
        self.history_keep_last_steps: int = 6
        self.max_new_tokens_cap: int = 2048
        self.enable_cuda_empty_cache: bool = True

        # 运行时历史
        self.history: List = []

    # ---------- 基础 ----------
    def _load_cfg_module(self, cfg_path: str):
        import importlib.util
        from pathlib import Path

        cfg_path = os.path.abspath(cfg_path)
        mod_name = Path(cfg_path).stem
        spec = importlib.util.spec_from_file_location(mod_name, cfg_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)  # type: ignore
        return module

    def is_ready(self) -> bool:
        return self.model is not None

    def initialize(self, cfg_path: str, save_dir: str, device_str: Optional[str] = None, force_reload: bool = False) -> str:
        """
        - 若已有模型且非强制重载，直接复用
        - 否则按 cfg 重建
        """
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
            except Exception:
                pass

        cfg = self._load_cfg_module(cfg_path)

        if device_str is not None:
            device = torch.device(device_str)
        else:
            device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        # 构建模型
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
        self._main_cfg_module = cfg
        self._device = device
        self._save_dir = save_dir
        self.history = []

        return f"✅ 模型已加载到 {device}，输出目录：{save_dir}"
    
    def update_sampling_config(self, mode: str) -> None:
        """
        根据 mode 加载 configs/ 下的配置文件，仅更新采样参数，不重载模型。
        """
        config_map = {
            "howto": "configs/example_config_visual_guidance.py",
            "story": "configs/example_config_visual_narrative.py",
            "t2i": "configs/example_config_t2i.py",
            "x2i": "configs/example_config_x2i.py",
            "default": "configs/config.py",
        }
        cfg_file = config_map.get(mode, "configs/config.py")
        if not os.path.exists(cfg_file):
            raise FileNotFoundError(f"配置文件 {cfg_file} 不存在！")

        cfg_module = self._load_cfg_module(cfg_file)

        # 只更新采样参数
        for key in self._sampling_keys:
            if hasattr(cfg_module, key):
                setattr(self.cfg_module, key, getattr(cfg_module, key))

    def clear_history(self) -> None:
        self.history = []

    def request_stop(self) -> None:
        self._stop_event.set()

    def reset_stop(self) -> None:
        self._stop_event.clear()

    def encode_and_set_prompt(self, sample: Dict[str, Any]) -> None:
        """
        保存用户输入的文本和图片，并编码prompt。
        """
        self.clear_history()
        seq = self.encode_prompt(sample)
        self.history.append(seq)

        # 保存用户输入和为本次会话创建目录
        user_dir = os.path.join(self._save_dir, "sessions")
        os.makedirs(user_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        idx = len(os.listdir(user_dir))
        session_dir = os.path.join(user_dir, f"session_{timestamp}_{idx:04d}")
        os.makedirs(session_dir, exist_ok=True)
        self._current_session_dir = session_dir  # 记录当前会话目录，供stream_events用

        # 保存文本
        text_prompt = sample.get("text_prompt", "")
        with open(os.path.join(session_dir, "prompt.txt"), "w", encoding="utf-8") as f:
            f.write(text_prompt)

        # 保存图片
        images = sample.get("images", [])
        for i, img_path in enumerate(images):
            try:
                img = Image.open(img_path)
                img.save(os.path.join(session_dir, f"image_{i:02d}.png"))
            except Exception:
                pass

    def encode_prompt(self, sample: Dict[str, Any]):
        # 你需要根据你的 pack_sample/sample 结构和 tokenizer 实现
        # 这里只是一个简单示例
        text_prompt = sample["text_prompt"]
        input_ids = self.tokenizer.encode(text_prompt, return_tensors="pt").to(self.model.device)
        return input_ids


    def _apply_memory_cfg_overrides(self, cfg_module: Any) -> None:
        """
        显存管理：根据配置文件动态调整显存相关参数。
        """
        try:
            if hasattr(cfg_module, "context_limit_tokens"):
                self.context_limit_tokens = int(cfg_module.context_limit_tokens)
            if hasattr(cfg_module, "history_keep_last_steps"):
                self.history_keep_last_steps = int(cfg_module.history_keep_last_steps)
            if hasattr(cfg_module, "max_new_tokens_cap"):
                self.max_new_tokens_cap = int(cfg_module.max_new_tokens_cap)
            if hasattr(cfg_module, "enable_cuda_empty_cache"):
                self.enable_cuda_empty_cache = bool(cfg_module.enable_cuda_empty_cache)
        except Exception:
            pass

    def _maybe_trim_history(self) -> None:
        """
        裁剪历史记录，避免显存溢出。
        """
        if not self.history:
            return

        # 保留最近的历史记录
        if len(self.history) > self.history_keep_last_steps:
            self.history = self.history[-self.history_keep_last_steps:]

        # 根据 token 限制进一步裁剪
        total_tokens = sum(len(seq) for seq in self.history)
        while total_tokens > self.context_limit_tokens and len(self.history) > 1:
            self.history.pop(0)
            total_tokens = sum(len(seq) for seq in self.history)

    def stream_events(self, max_rounds: int = 32, text_chunk_tokens: int = 64) -> Generator[Dict[str, Any], None, None]:
        """
        流式生成：优化文本和图片的流式输出。
        """
        assert self.model is not None and self.tokenizer is not None and self.vq_model is not None and self.cfg_module is not None

        # 获取 prompt
        if not self.history or len(self.history) == 0:
            raise RuntimeError("No prompt set. 请先调用 encode_and_set_prompt。")
        input_ids = self.history[-1]

        # 构造 unconditional_ids
        unc_prompt = getattr(self.cfg_module, "unc_prompt", "")
        unconditional_ids = self.tokenizer.encode(unc_prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)

        # full_unc_ids 可选
        if hasattr(self.cfg_module, "img_unc_prompt"):
            full_unc_ids = self.tokenizer.encode(self.cfg_module.img_unc_prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        else:
            full_unc_ids = None

        cfg = self.cfg_module
        force_same_image_size = True

        text_accum = ""
        session_dir = getattr(self, "_current_session_dir", self._save_dir)
        text_chunk_idx = 0
        image_chunk_idx = 0

        for result_tokens in generate(cfg, self.model, self.tokenizer, input_ids, unconditional_ids, full_unc_ids, force_same_image_size):
            if self._stop_event.is_set():
                break
            try:
                result = self.tokenizer.decode(result_tokens, skip_special_tokens=False)
                mm_out = multimodal_decode(result, self.tokenizer, self.vq_model)
                for item in mm_out:
                    if item[0] == "text":
                        # 分块输出文本
                        text_chunk = item[1][:text_chunk_tokens]
                        yield {"type": "text", "text": text_chunk}
                    elif item[0] == "image":
                        # 输出图片
                        img = item[1]
                        img_path = os.path.join(session_dir, f"gen_image_{image_chunk_idx:03d}.png")
                        img.save(img_path)
                        yield {"type": "image", "paths": [img_path]}
            except Exception as e:
                yield {"type": "text", "text": f"[ERROR] {e}"}
                break
        # 输出剩余文本（如有）
        if text_accum:
            text_path = os.path.join(session_dir, f"gen_text_{text_chunk_idx:03d}.txt")
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(text_accum)
            yield {"type": "text", "text": text_accum}

    @property
    def save_dir(self) -> str:
        assert self._save_dir is not None
        return self._save_dir


