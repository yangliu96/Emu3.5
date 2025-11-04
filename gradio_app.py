# app.py
import gradio as gr
import tempfile
import os
import shutil
import uuid
import re
from pathlib import Path
from typing import List, Dict, Any, Union

from PIL import Image, ImageOps
import importlib.util
import torch
import os.path as osp

from model_runtime import ModelRuntime

# ---------------- 基础 & 工具 ----------------

def _save_preview(img: Image.Image, max_hw: int = 768) -> str:
    """保存缩放预览，返回绝对路径（供 Chatbot 直接显示为图片）。"""
    preview = ImageOps.contain(img, (max_hw, max_hw)).convert("RGB")
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    preview.save(tmp.name, format="PNG")
    tmp.close()
    return os.path.abspath(tmp.name)

def _dup_path(src: str) -> str:
    """复制一个全新文件，避免同一路径在多条消息里复用导致的渲染问题。"""
    _, ext = os.path.splitext(src)
    tmp = tempfile.NamedTemporaryFile(suffix=ext or ".png", delete=False)
    tmp.close()
    shutil.copyfile(src, tmp.name)
    return os.path.abspath(tmp.name)

def _to_path_list(files: Union[List[str], List[Dict], None]) -> List[str]:
    """将 gr.Files 返回值统一成路径列表。"""
    paths: List[str] = []
    if not files:
        return paths
    for f in files:
        if isinstance(f, str):
            paths.append(f)
        elif isinstance(f, dict) and "name" in f:
            paths.append(f["name"])
        else:
            name = getattr(f, "name", None)
            if isinstance(name, str):
                paths.append(name)
    return paths

# ---------- 文本归档 + 中文友好分块 + 安全文本气泡 ----------

def _archive_text(text: str, folder: str = None) -> None:
    """
    取消写磁盘归档，避免按文件粒度导致前端错把每条当“独立消息”。
    如需排障，可在此改为环形内存日志或持久化开关。
    """
    return None

_CJK_SENT_SPLIT = re.compile(r'([。！？；!?;])')  # 捕获分隔符以便还原

def _split_sentences_cn_en(s: str) -> List[str]:
    """按中文/英文标点与换行切句，保留标点。"""
    if not s:
        return []
    paras = re.split(r'\n+', s)
    out = []
    for para in paras:
        if not para.strip():
            continue
        parts = _CJK_SENT_SPLIT.split(para)
        for i in range(0, len(parts), 2):
            sent = parts[i]
            punc = parts[i+1] if i+1 < len(parts) else ""
            chunk = (sent + punc).strip()
            if chunk:
                out.append(chunk)
    return out

def _chunk_text_cn_en(s: str, max_len: int = 80) -> List[str]:
    """
    先按句切，再把相邻句拼到不超过 max_len。
    对无空格长串做硬切，保证每块 <= max_len。
    """
    sents = _split_sentences_cn_en(s)
    if not sents:
        return []
    chunks, buf = [], ""
    for sent in sents:
        if len(buf) + len(sent) <= max_len:
            buf += (sent if not buf else sent)
        else:
            if buf:
                chunks.append(buf)
            if len(sent) > max_len:
                for i in range(0, len(sent), max_len):
                    piece = sent[i:i+max_len]
                    if piece:
                        chunks.append(piece)
                buf = ""
            else:
                buf = sent
    if buf:
        chunks.append(buf)
    return chunks

def _text_bubbles_safe(role: str, text: str, max_len: int = 80) -> List[gr.ChatMessage]:
    """
    把任意长文本 -> 多条安全短文本气泡。
    仅返回“内容文本”，不返回归档文件路径。
    每块前置一个零宽字符，进一步降低被误判为路径的概率。
    """
    msgs = []
    for chunk in _chunk_text_cn_en(text, max_len=max_len):
        msgs.append(gr.ChatMessage(role=role, content="\u2060" + chunk))
    return msgs

# // ---------------- 逐步产出：交由 ModelRuntime 管理 ----------------

# ---------------- 运行时封装（热重载/中断/裁剪） ----------------
_RUNTIME = ModelRuntime.instance()

def startup_initialize(cfg_path: str = "config/app_config.py",
                       save_dir: str = "./outputs",
                       device_str: str = None) -> str:
    return _RUNTIME.initialize(cfg_path=cfg_path, save_dir=save_dir, device_str=device_str)

def runtime_reload(cfg_path: str, save_dir: str, device_str: str) -> str:
    return _RUNTIME.initialize(cfg_path=cfg_path or "config/app_config.py",
                               save_dir=save_dir or "./outputs",
                               device_str=(device_str or None),
                               force_reload=True)

def runtime_clear_history() -> str:
    try:
        _RUNTIME.clear_history()
        return "🧹 已清空对话历史（仅模型态）。"
    except Exception as e:
        return f"清空历史失败：{e}"

def runtime_request_stop() -> str:
    _RUNTIME.request_stop()
    return "⏹️ 已请求停止当前生成。"

# ---------------- 打包 sample ----------------

VIS_TOKEN = "<|VIS_PLH|>"
SUP_START = "<|extra_100|>"
SUP_END   = "<|extra_101|>"

def pack_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    将 {text, images[]} 打包为:
    {
        "text_prompt": "You are a helpful assistant. USER: {text + VIS*n}ASSISTANT: <|extra_100|>",
        "visual_placeholder": "<|VIS_PLH|>",
        "supervised_start": "<|extra_100|>",
        "supervised_end": "<|extra_101|>",
        "image_list": [...]
    }
    - VIS_TOKEN 数量 == 图片数量
    - 若 text 内已有 VIS_TOKEN，先移除再按数量重插
    """
    text = (sample.get("text") or "").strip()
    images_in = sample.get("images") or []
    image_list: List[str] = [str(p) for p in images_in if p]
    n_vis = len(image_list)

    vis_re = re.compile(rf"\s*{re.escape(VIS_TOKEN)}\s*")
    text_wo_vis = vis_re.sub(" ", text).strip()
    text_wo_vis = re.sub(r"\s+", " ", text_wo_vis)

    if n_vis > 0:
        vis_suffix = " " + " ".join([VIS_TOKEN] * n_vis)
        user_text = (text_wo_vis + vis_suffix).strip()
    else:
        user_text = text_wo_vis

    mode = (sample.get("mode") or "").strip()
    if not mode or mode == "default":
        system_prefix = "You are a helpful assistant."
    if mode == "howto_new":
        system_prefix = "You are a helpful assistant for howto task. Please generate a response with interleaved text and images."
    else:
        system_prefix = f"You are a helpful assistant for {mode} task."
    print("system_prefix: {}".format(system_prefix))
    text_prompt = f"{system_prefix} USER: {user_text} ASSISTANT: {SUP_START}"
    return {
        "text_prompt": text_prompt,
        "visual_placeholder": VIS_TOKEN,
        "supervised_start": SUP_START,
        "supervised_end": SUP_END,
        "image_list": image_list,
    }

# ---------------- 回调（直接用已初始化的全局模型） ----------------

def on_submit(text: str, files: List[Any], mode: str, history: List[gr.ChatMessage]):
    text = (text or "").strip()
    file_paths = _to_path_list(files)

    # 左侧：用户文本（单条气泡，不再分块/归档）
    if text:
        history = history + [gr.ChatMessage(role="user", content="\u2060" + text)]
        yield history, gr.update(value=None), gr.update(value=None), history

    # 左侧：用户图片（逐张单独气泡）
    for p in file_paths:
        try:
            im = Image.open(p).convert("RGB")
            up = _save_preview(im, max_hw=768)
            history = history + [gr.ChatMessage(role="user", content=[up])]
            yield history, gr.update(value=None), gr.update(value=None), history
        except Exception:
            pass

    # 生成准备：重置 stop 标志，打包样本并设置到运行时
    _RUNTIME.reset_stop()
    raw_sample = {"text": text, "images": [os.path.abspath(p) for p in file_paths], "mode": (mode or "").strip()}
    sample = pack_sample(raw_sample)
    _RUNTIME.encode_and_set_prompt(sample)

    # 右侧：真正流式——事件驱动（text 小块即时、image 完整后即可上屏）
    acc_text = ""
    has_open_text = False
    for ev in _RUNTIME.stream_events(max_rounds=64, text_chunk_tokens=64):
        if ev.get("type") == "text":
            chunk = ev.get("text", "")
            if chunk:
                acc_text += chunk
                if not has_open_text or not history or history[-1].role != "assistant" or not isinstance(history[-1].content, str):
                    # 新建一个助手文本气泡
                    history = history + [gr.ChatMessage(role="assistant", content="\u2060" + acc_text)]
                    has_open_text = True
                else:
                    # 更新最后一个助手文本气泡
                    history = history[:-1] + [gr.ChatMessage(role="assistant", content="\u2060" + acc_text)]
                yield history, gr.update(value=None), gr.update(value=None), history
        elif ev.get("type") == "image":
            # 关闭当前文本气泡，后续文本新建
            has_open_text = False
            acc_text = ""
            for ip in ev.get("paths", []):
                echoed = _dup_path(ip)
                history = history + [gr.ChatMessage(role="assistant", content=[echoed])]
                yield history, gr.update(value=None), gr.update(value=None), history

def on_clear():
    _RUNTIME.reset_stop()
    try:
        _RUNTIME.clear_history()
    except Exception:
        pass
    return [], None, None, []

# ---------------- UI ----------------

with gr.Blocks(title="Model Text + Multi-Image (separate bubbles)") as demo:
    gr.Markdown("### 输入文本与多图；右侧按**生成步骤**依次输出：文本自动中文友好分块；支持 Stop/清空/热重载；显存友好。")

    chatbot = gr.Chatbot(type="messages", height=560, label="Conversation")

    with gr.Row():
        tb = gr.Textbox(
            label="Text",
            placeholder="Type something… (press Enter to send)",
            lines=2,
            autofocus=True,
            scale=5,
        )
        mode_dd = gr.Dropdown(
            label="System Prompt",
            choices=[
                "default",
                "lang",
                "vl",
                "t2i",
                "x2i",
                "howto",
                "story",
                "vla",
                "explore",
                "howto_new",
            ],
            value="default",
            scale=2,
        )
        files = gr.Files(
            label="Images (drop multiple here)",
            file_types=["image"],
            file_count="multiple",
            scale=5,
        )

    with gr.Row():
        send = gr.Button("Send", variant="primary")
        stop = gr.Button("Stop")
        clear = gr.Button("Clear")

    with gr.Accordion("高级设置（模型热重载 / 设备 / 输出目录）", open=False):
        with gr.Row():
            cfg_path_tb = gr.Textbox(label="Config Path", value="config/app_config.py", scale=4)
            save_dir_tb = gr.Textbox(label="Save Dir", value="./outputs", scale=3)
            device_tb = gr.Textbox(label="Device (e.g. cuda:0 / cpu)", value="", scale=2)
            reload_btn = gr.Button("Reload Model", variant="secondary", scale=1)

    state_history = gr.State([])  # 只存历史，可 deep copy
    ready_msg = gr.Markdown()
    demo.load(lambda: startup_initialize(), outputs=ready_msg)

    tb.submit(on_submit, inputs=[tb, files, mode_dd, state_history],
              outputs=[chatbot, tb, files, state_history])
    send.click(on_submit, inputs=[tb, files, mode_dd, state_history],
               outputs=[chatbot, tb, files, state_history])
    stop.click(lambda: runtime_request_stop(), outputs=[])
    clear.click(on_clear, outputs=[chatbot, tb, files, state_history])

    reload_btn.click(lambda cfg, sd, dev: runtime_reload(cfg, sd, dev),
                     inputs=[cfg_path_tb, save_dir_tb, device_tb],
                     outputs=ready_msg)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--port", type=int, default=None, help="Port for Gradio server")
    parser.add_argument("--host", type=str, default=None, help="Host for Gradio server (e.g. 0.0.0.0)")
    parser.add_argument("--cfg", type=str, default=None, help="Config path for model init (overrides UI)")
    parser.add_argument("--save_dir", type=str, default=None, help="Output directory for generations")
    parser.add_argument("--device", type=str, default=None, help="Device string, e.g. cuda:0 or cpu")
    args, _ = parser.parse_known_args()

    cfg_path = args.cfg or "config/app_config.py"
    save_dir = args.save_dir or "./outputs"
    device_str = args.device or None

    print(startup_initialize(cfg_path=cfg_path, save_dir=save_dir, device_str=device_str))

    launch_kwargs = {}
    if args.port is not None:
        launch_kwargs["server_port"] = args.port
    if args.host is not None:
        launch_kwargs["server_name"] = args.host
    demo.launch(**launch_kwargs)