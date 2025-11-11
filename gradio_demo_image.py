# -*- coding: utf-8 -*-
import argparse
import gradio as gr
import tempfile
import os
import shutil
from model_runtime import ModelRuntime

_RUNTIME = ModelRuntime.instance()

CSS = """
/* 整个聊天区域 */
.chatbot {
    max-height: 540px;
}

/* user 消息靠右显示 */
.chatbot .message.user {
    background: #dff7e6 !important;
    margin-left: auto !important;
    text-align: right !important;
    border-radius: 12px 12px 2px 12px !important;
}

/* assistant 消息靠左显示 */
.chatbot .message.assistant {
    background: #eef2ff !important;
    margin-right: auto !important;
    text-align: left !important;
    border-radius: 12px 12px 12px 2px !important;
}

/* 去掉 user / assistant label */
.chatbot .message .label {
    display: none !important;
}
"""

# ===================== NEW: 纵横比映射与解析 =====================
aspect_ratios = {
    "4:3": "55*73",
    "21:9": "41*97",
    "16:9": "47*85",
    "3:2": "52*78",
    "1:1": "64*64",
    "3:4": "73*55",
    "9:16": "85*47",
    "2:3": "78*52",
    "auto": None,
}

def get_target_size(aspect_ratio: str):
    value = aspect_ratios.get(aspect_ratio, None)
    if value is None:
        return None, None
    h, w = map(int, value.split("*"))
    return h, w
# ================================================================

def _dup_path(src: str) -> str:
    """复制一个全新文件，避免同一路径在多条消息里复用导致的渲染问题。"""
    _, ext = os.path.splitext(src)
    tmp = tempfile.NamedTemporaryFile(suffix=ext or ".png", delete=False)
    tmp.close()
    shutil.copyfile(src, tmp.name)
    return os.path.abspath(tmp.name)

def startup_initialize(cfg_path: str, save_dir: str, device_str: str | None = None):
    return _RUNTIME.initialize(cfg_path=cfg_path, save_dir=save_dir, device_str=device_str)

# ===================== MOD: 增加 aspect_ratio / target_size 传入 =====================
def on_submit(text, files, mode, aspect_ratio, history):
    # 计算目标尺寸（仅 t2i 生效；x2i 传 None）

    tgt_h, tgt_w = (get_target_size(aspect_ratio) if mode == "t2i" else (None, None))
    _RUNTIME.update_sampling_config(mode=mode, target_height=tgt_h, target_width=tgt_w)  # 可能被忽略


    # FIX: gr.Files(..., type="filepath") 返回的是路径字符串列表，不是对象；不要用 f.name
    image_paths = files or []

    # 把尺寸也一并塞进 sample，便于后端 encode 时读取（若后端暂时不用，也没关系）
    sample = {
        "text": text,
        "images": image_paths,            # FIX: 直接用路径字符串
        "target_size": (tgt_h, tgt_w),    # NEW: 传入尺寸
        "aspect_ratio": aspect_ratio,     # NEW: 记录所选纵横比
    }
    _RUNTIME.encode_and_set_prompt(sample)

    # 用户消息
    if image_paths:
        history.append({"role": "user", "content": text})
        history.append({"role": "user", "content": image_paths})
    else:
        history.append({"role": "user", "content": text})
    yield history, "", None, history

    # 占位 assistant 消息
    assistant_acc = ""
    history.append({"role": "assistant", "content": assistant_acc})
    yield history, "", None, history

    # Streaming
    for ev in _RUNTIME.stream_events(text_chunk_tokens=64):
        if ev["type"] == "text":
            assistant_acc += ev["text"]
            history[-1] = {"role": "assistant", "content": assistant_acc}
            yield history, "", None, history

        elif ev["type"] == "image":
            for ip in ev.get("paths", []):
                echoed = _dup_path(ip)
                history.append({"role": "assistant", "content": [echoed]})
                yield history, gr.update(value=None), gr.update(value=None), history

            assistant_acc = ""
            history.append({"role": "assistant", "content": assistant_acc})
# =================================================================

def clear_chat():
    _RUNTIME.history.clear()
    return [], []

def on_stop():
    _RUNTIME.request_stop()
    return "🛑 已发送停止信号（本轮生成将尽快结束显示）"

def build_ui():
    with gr.Blocks(css=CSS) as demo:
        gr.Markdown("# 🦄 Emu 3.5-Image Gradio Demo")

        with gr.Row():
            with gr.Column(scale=6):
                chat = gr.Chatbot(
                    label="Chat",
                    height=540,
                    elem_classes="chatbot",
                    type="messages",
                )
                state = gr.State([])

                mode = gr.Dropdown(
                    label="Generation Mode",
                    choices=["t2i", "x2i"],
                    value="t2i"
                )

                # ===================== NEW: 纵横比选项（仅 t2i 使用） =====================
                aspect_ratio = gr.Dropdown(
                    label="Aspect Ratio (T2I)",
                    choices=list(aspect_ratios.keys()),
                    value="auto",
                    interactive=True,
                    visible=True,  # 初始 value 为 t2i，因此可见
                )

                # 根据 mode 切换纵横比控件显隐
                def _toggle_ar(m):
                    return gr.update(visible=(m == "t2i"))
                mode.change(_toggle_ar, inputs=[mode], outputs=[aspect_ratio])
                # ========================================================================

                text = gr.Textbox(label="💬 Prompt", placeholder="Enter your prompt...", lines=2)
                # FIX: 使用 filepath，on_submit 里按路径字符串处理
                files = gr.Files(label="📷 Upload image(s)", file_count="multiple", type="filepath")

                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    stop = gr.Button("Stop")
                    clear = gr.Button("Clear")
                
                status = gr.Markdown("")

        # 绑定：把 aspect_ratio 也作为输入传入 on_submit
        send.click(
            on_submit,
            inputs=[text, files, mode, aspect_ratio, state],   # NEW: 多了 aspect_ratio
            outputs=[chat, text, files, state]
        )

        stop.click(on_stop, outputs=[status])
        clear.click(clear_chat, outputs=[chat, state])

    return demo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    args.cfg = "configs/example_config_t2i.py"
    args.save_dir = "./outputs"
    args.device = None
    print(startup_initialize(args.cfg, args.save_dir, args.device))
    ui = build_ui()
    ui.queue()
    ui.launch(
        server_name=args.host,
        server_port=args.port,
        # show_error=True,
        # prevent_thread_lock=True,
        # allowed_paths=["."],
        # enable_monitoring=False,
    )

if __name__ == "__main__":
    main()