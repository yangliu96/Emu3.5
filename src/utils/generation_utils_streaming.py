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
    图片：等到完整 <|BOI|>...<|EOI|> 片段生成完毕后一次性输出
    最终：返回 token ids（与非流式保持一致，便于上游落盘/可视化）
    """
    input_ids_len = input_ids.shape[1]
    logits_processor, generation_config = _build_generation_objects(
        cfg, model, tokenizer, unconditional_ids, full_unconditional_ids, force_same_image_size
    )

    # 若无法使用 streamer，则退化为非流式但不报错
    if not _HAS_STREAMER or tokenizer is None:
        gen_ids = non_streaming_generate(
            cfg, model, tokenizer, input_ids, unconditional_ids,
            full_unconditional_ids, force_same_image_size
        )
        # 最终 ids
        yield {"type": "final_ids", "ids": gen_ids}
        # 尝试解析完整图片并返回
        try:
            decoded = tokenizer.batch_decode(
                torch.tensor(np.concatenate([input_ids.cpu(), torch.tensor([gen_ids])], axis=1))
                if isinstance(input_ids, np.ndarray) else
                torch.cat([input_ids, torch.tensor([gen_ids], device=input_ids.device)], dim=1),
                skip_special_tokens=False
            )[0]
            for kind, payload in multimodal_decode(decoded, tokenizer, getattr(cfg, "vision_tokenizer", None)):
                if kind == "image" and isinstance(payload, Image.Image):
                    yield {"type": "image", "image": payload}
        except Exception:
            pass
        return

    # --- 使用 TextIteratorStreamer 实现文本流式 ---
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,  # 仅流文本，不把视觉 token/特殊符号打到前端
    )

    out_holder: Dict[str, Any] = {}

    def _worker():
        # 生成并记录完整 ids，供收尾阶段解码图片/返回 ids
        tokens = model.generate(
            input_ids,
            generation_config,
            logits_processor=logits_processor,
        )
        # tokens: [bsz, total_len]
        out_holder["token_ids"] = tokens

        # 将解码文本写入 streamer（方式：再次 decode 增量或直接整体 decode？
        # 这里借助 streamer 需要在 generate 时就传入。为了保持与官方 generate 签名一致，
        # 我们采用“二段式”：先 run generate 拿 ids，后面用增量解码模拟流式。
        # 为了尽量贴近“流”，我们对新生成部分做小片段推送。
        gen_part = tokens[:, input_ids_len:]  # [1, new_len]
        # 用 tokenizer 逐小段 decode，构造“近似流”
        step = max(1, int(getattr(cfg, "stream_step_tokens", 16)))
        pieces = []
        for i in range(0, gen_part.shape[1], step):
            sub = gen_part[:, : i + step]
            text = tokenizer.batch_decode(sub, skip_special_tokens=True)[0]
            pieces.append(text)
        # 把差分推给 streamer
        last = ""
        for cur in pieces:
            delta = cur[len(last):]
            if delta:
                streamer.put(delta)
                last = cur
        streamer.end()

    # 后台线程生成
    th = threading.Thread(target=_worker, daemon=True)
    th.start()

    # 主线程持续从 streamer 取文本片段
    for piece in streamer:
        if piece:
            yield {"type": "text", "text": piece}

    # 生成结束：返回最终 ids，并且如果包含完整图片，在此时一次性输出
    tokens = out_holder.get("token_ids", None)
    if tokens is None:
        return  # 理论上不会发生

    gen_token_ids = tokens[:, input_ids_len:]
    if isinstance(gen_token_ids, torch.Tensor):
        gen_np = gen_token_ids[0].detach().cpu().numpy()
    else:
        gen_np = np.array(gen_token_ids[0], dtype=np.int64)

    # 最终 ids
    yield {"type": "final_ids", "ids": gen_np}

    # 图片一次性输出：将全序列 decode 后抽取 <|BOI|>...<|EOI|>
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