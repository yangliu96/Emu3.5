import os
import threading
from typing import Any, Dict, Generator, List, Optional, Tuple

import torch

from multi_turn_emu3p5 import MultiTurnEmu3Generator


class ModelRuntime:
    """
    统一管理模型实例、上下文与流式生成；支持：
    - 初始化/热重载（不重启 App）
    - 清空历史/选择性裁剪历史（显存保护）
    - 生成中断（Stop）
    - 分步产出：文本流式分块 + 图像就绪即上传
    """

    _singleton: Optional["ModelRuntime"] = None

    @classmethod
    def instance(cls) -> "ModelRuntime":
        if cls._singleton is None:
            cls._singleton = ModelRuntime()
        return cls._singleton

    def __init__(self) -> None:
        self._generator: Optional[MultiTurnEmu3Generator] = None
        self._cfg_module: Optional[Any] = None
        self._device: Optional[torch.device] = None
        self._save_dir: Optional[str] = None
        self._stop_event = threading.Event()

        # 显存/上下文保护参数（可由 cfg 覆盖）
        self.context_limit_tokens: int = 8192  # 超过约 8k 可能爆显存
        self.history_keep_last_steps: int = 6   # 保留最近 N 段（除第一条用户指令）
        self.max_new_tokens_cap: int = 2048     # 单次生成上限
        self.enable_cuda_empty_cache: bool = True

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
        return self._generator is not None

    def initialize(self, cfg_path: str, save_dir: str, device_str: Optional[str] = None, force_reload: bool = False) -> str:
        """
        - 若已有模型且非强制重载，直接复用
        - 否则按 cfg 重建
        """
        if self._generator is not None and not force_reload:
            return "✅ 模型已就绪（复用现有实例）"

        # 卸载旧模型
        if self._generator is not None:
            try:
                self._generator = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        cfg = self._load_cfg_module(cfg_path)

        if device_str is not None:
            device = torch.device(device_str)
        else:
            device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        generator = MultiTurnEmu3Generator(cfg=cfg, device=device)

        save_dir = os.path.abspath(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        # 接入显存保护的参数（若配置中存在则覆盖默认）
        self._apply_memory_cfg_overrides(cfg)

        # 绑定
        self._generator = generator
        self._cfg_module = cfg
        self._device = device
        self._save_dir = save_dir

        # 限制 max_new_tokens，避免过大
        try:
            max_cap = int(self.max_new_tokens_cap)
            generator.generation_config.max_new_tokens = min(
                int(generator.generation_config.max_new_tokens or max_cap), max_cap
            )
        except Exception:
            pass

        return f"✅ 模型已加载到 {device}，输出目录：{save_dir}"

    # ---------- 显存/上下文 ----------
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
        except Exception:
            pass

    def _maybe_trim_history(self) -> None:
        """
        保持 history 不超过 context_limit_tokens：
        - 永远保留第一条用户输入（history[0]）
        - 从第二条开始裁剪到最近 history_keep_last_steps 条
        - 若仍超限，则进一步从第二条开始向后裁到满足 token 上限
        """
        g = self._generator
        if g is None or not g.history:
            return

        # 先做基于步数的截断
        if len(g.history) - 1 > self.history_keep_last_steps:
            g.history = g.history[:1] + g.history[-self.history_keep_last_steps :]

        # 再做基于 token 的精细截断
        total = g.get_history_length()
        if total <= self.context_limit_tokens:
            return

        # 保留第一条，向后丢弃直到满足上限
        kept = [g.history[0]]
        for seq in reversed(g.history[1:]):
            kept.insert(1, seq)
            # 重新计算
            tmp_len = sum(len(s) for s in kept)
            if tmp_len > self.context_limit_tokens:
                kept.pop(1)
                break
        g.history = kept

    # ---------- 控制 ----------
    def clear_history(self) -> None:
        if self._generator is not None:
            self._generator.reset_history()

    def request_stop(self) -> None:
        self._stop_event.set()

    def reset_stop(self) -> None:
        self._stop_event.clear()

    # ---------- 生成流 ----------
    def encode_and_set_prompt(self, sample: Dict[str, Any]) -> None:
        assert self._generator is not None
        self._generator.reset_history()
        seq = self._generator.encode_prompt(sample)
        self._generator.append_to_history(seq)

    def stream_events(self, max_rounds: int = 32, text_chunk_tokens: int = 64) -> Generator[Dict[str, Any], None, None]:
        """
        真正的流式：
        - 文本：按小块 token 流式产出（T 段），每小块立即 yield {type: "text", text: chunk}
        - 图片：检测到进入图片段后（I 段）一次性生成完该图并 yield {type: "image", paths: [...]}。
        - 可随时 request_stop()
        """
        assert self._generator is not None and self._cfg_module is not None and self._save_dir is not None

        import os.path as osp

        self._maybe_trim_history()

        # 为本轮生成建立 case 目录
        base = osp.join(self._save_dir, "one_step_gen_samples")
        os.makedirs(base, exist_ok=True)
        existing = []
        for name in os.listdir(base):
            if name.startswith("case") and name[4:].isdigit():
                existing.append(int(name[4:]))
        next_id = max(existing) + 1 if existing else 1
        gen_path = osp.join(base, f"case{next_id}")
        os.makedirs(gen_path, exist_ok=True)

        gen = self._generator
        image_start_id = gen.cfg.special_token_ids.image_start_token

        # 保护原始 max_new_tokens，T 段用较小块做近似流式
        original_max_new = int(getattr(gen.generation_config, "max_new_tokens", self.max_new_tokens_cap))

        rounds = 0
        while not self._stop_event.is_set() and rounds < max_rounds:
            # 先进入 T 段：小块生成并流式发送
            reached_image = False
            t_accum = None  # 累积本步T段tokens，用于与I段拼接后再解码图片
            while not self._stop_event.is_set():
                gen.generation_config.max_new_tokens = min(int(text_chunk_tokens), int(self.max_new_tokens_cap))
                outputs, stop_flag = gen.generate_one_step_0724(mode='T')

                if outputs is None or len(outputs) == 0:
                    break

                # 是否触发了图像开始（输出中包含 image_start）
                if (outputs == image_start_id).any():
                    reached_image = True

                # 直接对新增 tokens 解码（跳过特殊符号）
                try:
                    text_chunk = gen.tokenizer.decode(outputs, skip_special_tokens=False)
                except Exception:
                    text_chunk = ""
                # text_chunk = (text_chunk or "").strip()
                if text_chunk:
                    yield {"type": "text", "text": text_chunk}

                # 累积当前T段tokens
                try:
                    import numpy as np
                    t_accum = outputs if t_accum is None else np.concatenate([t_accum, outputs])
                except Exception:
                    t_accum = outputs

                if stop_flag:
                    # 终止（eos）
                    gen.generation_config.max_new_tokens = original_max_new
                    return

                if reached_image:
                    break

                if self.enable_cuda_empty_cache and torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

            # 如未进入图片，直接下一轮
            if not reached_image:
                rounds += 1
                continue

            # 进入 I 段：一次性把图片生成出来（注意：必须包含上一步的T段，否则图片解析会失衡）
            if self._stop_event.is_set():
                break

            # 给 I 段更大的上限，以保证一张图完整生成
            gen.generation_config.max_new_tokens = int(self.max_new_tokens_cap)
            outputs, stop_flag = gen.generate_one_step_0724(mode='I')
            if outputs is not None and len(outputs) > 0:
                # 规范化：确保序列恰好是 [text(without <image start>)] + [<image start> + image_tokens...]
                import numpy as np
                text_part = t_accum if t_accum is not None else np.array([], dtype=outputs.dtype)
                # 去除 T 段内的所有 image_start，防止前置残留
                text_part = text_part[text_part != image_start_id]
                # I 段若不以 image_start 开头，则补齐
                i_part = outputs
                if i_part[0] != image_start_id:
                    i_part = np.concatenate([np.array([image_start_id], dtype=outputs.dtype), i_part])
                total_seq = np.concatenate([text_part, i_part])

                result = gen.split_seq_by_image(total_seq)
                # 优先按一步[text,image]对齐；若断言失败，回退到自动对齐
                try:
                    steps, step_imgs = gen.decode_generated_results(result, gen_path, step_idx=rounds)
                except AssertionError:
                    steps, step_imgs = gen.decode_generated_results(result, gen_path)
                imgs_path = step_imgs[0] if step_imgs else []
                imgs_path = [os.path.abspath(p) for p in imgs_path]
                if imgs_path:
                    yield {"type": "image", "paths": imgs_path}

            if self.enable_cuda_empty_cache and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            if stop_flag:
                gen.generation_config.max_new_tokens = original_max_new
                return

            rounds += 1

        # 恢复上限
        gen.generation_config.max_new_tokens = original_max_new

    # ---------- 便捷读取 ----------
    @property
    def generator(self) -> MultiTurnEmu3Generator:
        assert self._generator is not None
        return self._generator

    @property
    def cfg(self) -> Any:
        assert self._cfg_module is not None
        return self._cfg_module

    @property
    def save_dir(self) -> str:
        assert self._save_dir is not None
        return self._save_dir


