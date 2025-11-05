# -*- coding: utf-8 -*-
# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import re
import threading
import time
from typing import Generator, List, Tuple, Dict, Any, Optional

from PIL import Image
import numpy as np
import torch
from transformers import GenerationConfig

# 官方依赖：自定义 logits processor（保持不变）
from transformers.generation import LogitsProcessorList
from .logits_processor import (
    UnbatchedClassifierFreeGuidanceLogitsForVisualTokenWithDifferentialTopKProcessor
)

# ---- 仅在可用时启用文本流式 ----
try:
    from transformers import TextIteratorStreamer
    _HAS_STREAMER = True
except Exception:
    _HAS_STREAMER = False


@torch.no_grad()
def generate(
    cfg,
    model,
    tokenizer,
    input_ids,
    unconditional_ids,
    full_unconditional_ids=None,
    force_same_image_size=True,
):
    """
    行为与官方一致：
      - cfg.streaming=False: 一次性返回最终 token ids（np.ndarray）
      - cfg.streaming=True : 逐步 yield 文本片段 {"type": "text", "text": str}
                             结束时 yield 最终 ids {"type": "final_ids", "ids": np.ndarray}
                             若包含完整图片，最后追加 {"type": "image", "image": PIL.Image.Image}
    """

    if getattr(cfg, "streaming", False):
        # 官方原注释：yield from streaming_generate(...)
        # 这里补全真实实现（见下）
        yield from streaming_generate(
            cfg, model, tokenizer, input_ids, unconditional_ids,
            full_unconditional_ids=full_unconditional_ids,
            force_same_image_size=force_same_image_size
        )
    else:
        # 原始一次性生成（保持不变）
        yield non_streaming_generate(
            cfg, model, tokenizer, input_ids, unconditional_ids,
            full_unconditional_ids, force_same_image_size
        )


def _build_generation_objects(
    cfg, model, tokenizer, unconditional_ids, full_unconditional_ids, force_same_image_size
):
    """复用官方构造流程，避免行为漂移。"""
    logits_processor = LogitsProcessorList()
    logits_processor.append(
        build_logits_processor(
            cfg, unconditional_ids, model, tokenizer,
            full_unconditional_ids, force_same_image_size=force_same_image_size
        )
    )
    generation_config = GenerationConfig(
        **cfg.sampling_params,
        pad_token_id=cfg.special_token_ids["PAD"],
        eos_token_id=cfg.special_token_ids["EOS"],
    )
    return logits_processor, generation_config


def streaming_generate(
    cfg,
    model,
    tokenizer,
    input_ids,
    unconditional_ids,
    full_unconditional_ids=None,
    force_same_image_size=True,
):
    """
    文本：流式输出（多次 yield）
    图片：生成时即时输出，文本和图片交替输出
    最终：返回 token ids（与非流式保持一致，便于上游落盘/可视化）
    """
    import re

    input_ids_len = input_ids.shape[1]
    logits_processor, generation_config = _build_generation_objects(
        cfg, model, tokenizer, unconditional_ids, full_unconditional_ids, force_same_image_size
    )

    if not _HAS_STREAMER or tokenizer is None:
        gen_ids = non_streaming_generate(
            cfg, model, tokenizer, input_ids, unconditional_ids,
            full_unconditional_ids, force_same_image_size
        )
        yield {"type": "final_ids", "ids": gen_ids}
        try:
            decoded = tokenizer.batch_decode(
                torch.tensor(np.concatenate([input_ids.cpu(), torch.tensor([gen_ids])], axis=1))
                if isinstance(input_ids, np.ndarray)
                else torch.cat([input_ids, torch.tensor([gen_ids], device=input_ids.device)], dim=1),
                skip_special_tokens=False,
            )[0]
            for kind, payload in multimodal_decode(decoded, tokenizer, getattr(cfg, "vision_tokenizer", None)):
                if kind == "image" and isinstance(payload, Image.Image):
                    yield {"type": "image", "image": payload}
        except Exception:
            pass
        return

    # --- 初始化流式输出 ---
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=False,  # 保留特殊符号以便我们自己处理
    )

    out_holder: Dict[str, Any] = {}

    def _worker():
        tokens = model.generate(
            input_ids,
            generation_config,
            logits_processor=logits_processor,
            streamer=streamer,
        )
        out_holder["token_ids"] = tokens

    th = threading.Thread(target=_worker, daemon=True)
    th.start()

    # --- 处理文本输出 ---
    _img_start_re = re.compile(r"(?:<\|\s*(image\s*start|imagestart|boi)\s*\|>)", re.IGNORECASE)
    _img_end_re = re.compile(r"(?:<\|\s*(image\s*end|imageend|eoi)\s*\|>)", re.IGNORECASE)
    _global_cot_re = re.compile(r"<\|extra_60\|>(.*?)<\|extra_61\|>", re.DOTALL)
    _step_cot_re = re.compile(r"<\|extra_50\|>(.*?)<\|extra_51\|>", re.DOTALL)
    _special_tok_re = re.compile(r"<\|[^|>]+\|>")

    buffer = ""
    image_mode = False
    image_tokens = []  # 缓存图片 token，等到最后一次性解码
    image_buffer = ""  # 用来缓存图片 token，跨多个 buffer 时不会丢失

    def _emit_clean_text(txt: str):
        """输出 Global COT / Step COT / 普通文本，并保持换行"""
        # 查找并替换 Global COT
        for m in list(_global_cot_re.finditer(txt)):
            yield {"type": "text", "text": f"[Global COT] {m.group(1).strip()}\n"}
        txt = _global_cot_re.sub("", txt)

        # 查找并替换 Step COT
        for m in list(_step_cot_re.finditer(txt)):
            yield {"type": "text", "text": f"[Step COT] {m.group(1).strip()}\n"}
        txt = _step_cot_re.sub("", txt)

        # 去除特殊 token
        clean_txt = _special_tok_re.sub("", txt).strip()
        if clean_txt:
            clean_txt = clean_txt.replace("<|newline|>", "\n")  # 保留换行符
            yield {"type": "text", "text": clean_txt}

    # --- 主循环：逐片解析 ---
    for piece in streamer:
        if not piece:
            continue
        buffer += piece

        while True:
            if image_mode:
                mend = _img_end_re.search(buffer)
                if mend:
                    # 将图片 token 收集完后，通过 tokenizer 解码
                    image_tokens.append(image_buffer + buffer[:mend.end()])
                    buffer = buffer[mend.end():]
                    image_mode = False

                    # 解码图片并输出
                    image_token_str = "".join(image_tokens)
                    import pdb; pdb.set_trace()
                    try:
                        image = multimodal_decode(image_token_str, tokenizer, getattr(cfg, "vision_tokenizer", None))[0][1]
                        yield {"type": "image", "image": image}
                    except Exception:
                        pass
                    image_tokens = []  # 重置图片 token 缓存
                    image_buffer = ""  # 清空图片缓存
                    continue
                else:
                    # 图片 token 不完整，继续缓存
                    image_buffer += buffer
                    buffer = ""  # 清空当前 buffer，等待更多图片 token
                    break
            else:
                mstart = _img_start_re.search(buffer)
                if mstart:
                    pre = buffer[:mstart.start()]
                    for ev in _emit_clean_text(pre):
                        yield ev
                    buffer = buffer[mstart.end():]
                    image_mode = True
                    continue
                else:
                    stable, buffer = buffer[:-256], buffer[-256:]  # 增加缓存大小
                    if stable:
                        for ev in _emit_clean_text(stable):
                            yield ev
                    break

    # --- 最后 flush 缓冲 ---
    if not image_mode and buffer:
        for ev in _emit_clean_text(buffer):
            yield ev
    buffer = ""

    # --- 输出最终 tokens ---
    tokens = out_holder.get("token_ids", None)
    if tokens is None:
        return
    gen_token_ids = tokens[:, input_ids_len:]
    gen_np = (
        gen_token_ids[0].detach().cpu().numpy()
        if isinstance(gen_token_ids, torch.Tensor)
        else np.array(gen_token_ids[0], dtype=np.int64)
    )
    yield {"type": "final_ids", "ids": gen_np}

    # --- 解码图片 ---
    try:
        full_text = tokenizer.batch_decode(tokens, skip_special_tokens=False)[0]
        for kind, payload in multimodal_decode(full_text, tokenizer, getattr(cfg, "vision_tokenizer", None)):
            if kind == "image" and isinstance(payload, Image.Image):
                yield {"type": "image", "image": payload}
    except Exception:
        pass


def non_streaming_generate(
    cfg,
    model,
    tokenizer,
    input_ids,
    unconditional_ids,
    full_unconditional_ids=None,
    force_same_image_size=True,
):
    input_ids_len = input_ids.shape[1]
    logits_processor = LogitsProcessorList()
    logits_processor.append(
        build_logits_processor(
            cfg,
            unconditional_ids,
            model,
            tokenizer,
            full_unconditional_ids,
            force_same_image_size=force_same_image_size,
        )
    )
    generation_config = GenerationConfig(
        **cfg.sampling_params,
        pad_token_id=cfg.special_token_ids["PAD"],
        eos_token_id=cfg.special_token_ids["EOS"],
    )
    token_ids = model.generate(
        input_ids,
        generation_config,
        logits_processor=logits_processor,
    )
    gen_token_ids = token_ids[:, input_ids_len:]
    return gen_token_ids[0].detach().cpu().numpy()


def build_logits_processor(
    cfg,
    unconditional_ids,
    model,
    tokenizer,
    full_unconditional_ids=None,
    force_same_image_size=True,
):
    logits_processor = UnbatchedClassifierFreeGuidanceLogitsForVisualTokenWithDifferentialTopKProcessor(
        guidance_scale=cfg.classifier_free_guidance,
        unconditional_ids=unconditional_ids,
        full_unconditional_ids=full_unconditional_ids,
        model=model,
        tokenizer=tokenizer,
        unconditional_type=cfg.unconditional_type,
        target_height=getattr(cfg, "target_height", None),
        target_width=getattr(cfg, "target_width", None),
        image_cfg_scale=getattr(cfg, "image_cfg_scale", 1.0),
        use_differential_sampling=cfg.sampling_params["use_differential_sampling"],
        text_top_k=cfg.sampling_params["text_top_k"],
        text_top_p=cfg.sampling_params["text_top_p"],
        text_temperature=cfg.sampling_params["text_temperature"],
        image_top_k=cfg.sampling_params["image_top_k"],
        image_top_p=cfg.sampling_params["image_top_p"],
        image_temperature=cfg.sampling_params["image_temperature"],
        force_same_image_size=force_same_image_size,
    )
    return logits_processor


@torch.no_grad()
def multimodal_decode(
    outputs,
    tokenizer,
    vision_tokenizer,
):
    outputs = outputs.replace("<|extra_101|>", "").replace("<|extra_204|>", "")
    pattern = re.compile(
        rf"({re.escape(tokenizer.bog_token)}.*?{re.escape(tokenizer.eog_token)}|"
        rf"{re.escape(tokenizer.boc_token)}.*?{re.escape(tokenizer.eoc_token)}|"
        rf"{re.escape(tokenizer.boi_token)}.*?{re.escape(tokenizer.eoi_token)})",
        re.DOTALL,
    )
    multimodal_output = []
    chunks = re.split(pattern, outputs)
    for c in chunks:
        if len(c) == 0:
            continue
        if tokenizer.boi_token in c and tokenizer.eoi_token in c:
            image = decode_image(c, tokenizer, vision_tokenizer)
            if image is not None:
                multimodal_output.append(("image", image))
        elif tokenizer.bog_token in c and tokenizer.eog_token in c:
            multimodal_output.append(
                ("global_cot", c.replace(tokenizer.bog_token, "").replace(tokenizer.eog_token, ""))
            )
        elif tokenizer.boc_token in c and tokenizer.eoc_token in c:
            multimodal_output.append(
                ("image_cot", c.replace(tokenizer.boc_token, "").replace(tokenizer.eoc_token, ""))
            )
        # exclude incomplete image
        elif tokenizer.boi_token not in c and len(c.strip()) > 0:
            multimodal_output.append(("text", c))
    return multimodal_output


def decode_image(image_string, tokenizer, vision_tokenizer):
    image: List[List[int]] = []
    image_rows = re.split(re.escape(tokenizer.eol_token), image_string)
    for r in image_rows:
        token_ids = re.findall(r"<\|visual token (\d+)\|>", r)
        if len(token_ids) > 0:
            row_token = [int(m) for m in token_ids]
            image.append(row_token)
    try:
        image = torch.tensor(
            image, dtype=torch.long, device=next(iter(vision_tokenizer.parameters())).device
        )
        h, w = image.shape
        image = vision_tokenizer.decode_code(image[None], shape=(1, h, w, 256)).float()
        image = image[0].permute(1, 2, 0)
        image = Image.fromarray(
            ((image + 1.0) * 127.5).clamp(0, 255).detach().cpu().numpy().astype(np.uint8)
        )
        return image
    except Exception as ex:
        print(f"decode image failed {ex}")
        return None