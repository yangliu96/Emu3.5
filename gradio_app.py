# -*- coding: utf-8 -*-
import argparse
import gradio as gr
from model_runtime import ModelRuntime

_RUNTIME = ModelRuntime.instance()

CSS = """
.chatbot .message.user {
    background: #DCF8C6 !important;
    color: #111;
}
.chatbot .message.assistant {
    background: #E8EBFF !important;
    color: #111;
}
"""

def startup_initialize(cfg_path: str, save_dir: str, device_str: str = None):
    return _RUNTIME.initialize(cfg_path=cfg_path, save_dir=save_dir, device_str=device_str)


def on_submit(text, files, mode, history):
    """点击 send"""
    _RUNTIME.update_sampling_config(mode)
    sample = {"text": text, "images": [f.name for f in files] if files else []}
    _RUNTIME.encode_and_set_prompt(sample)

    # ✅ UI 显示用户输入（文本 + 图片）
    chat_entry = {"role": "user", "content": []}
    if text:
        chat_entry["content"].append({"type": "text", "text": text})
    if files:
        chat_entry["content"].append({"type": "image", "path": [f.name for f in files]})

    history.append(chat_entry)

    yield history, "", None, history  # 清空输入框

    assistant_msg = {"role": "assistant", "content": []}
    history.append(assistant_msg)

    # ✅ Streaming 处理 generate() 输出
    for ev in _RUNTIME.stream_events(text_chunk_tokens=48):

        if ev["type"] == "text":
            assistant_msg["content"].append({"type": "text", "text": ev["text"]})
            history[-1] = assistant_msg
            yield history, "", None, history

        elif ev["type"] == "image":
            assistant_msg["content"].append({"type": "image", "path": ev["paths"]})
            history[-1] = assistant_msg
            yield history, "", None, history


def clear_chat():
    _RUNTIME.history.clear()
    return [], []


def on_stop():
    _RUNTIME.request_stop()
    return "🛑 正在停止..."


def build_ui():
    with gr.Blocks(css=CSS) as demo:
        gr.Markdown("# 🦄 Emu 3.5 (BAAI) Streaming Demo")

        with gr.Row():
            with gr.Column(scale=2):
                cfg = gr.Textbox(label="🧩 Config Path", value="configs/config.py")
                save_dir = gr.Textbox(label="📁 Output Directory", value="./outputs")
                device = gr.Textbox(label="⚙️ Device", value="cuda:0")
                mode = gr.Dropdown(
                    label="Generation Mode",
                    choices=["default", "howto", "story", "t2i", "x2i"],
                    value="default"
                )
                init_btn = gr.Button("🚀 Load Model", variant="primary")
                status = gr.Markdown("")

            with gr.Column(scale=6):
                chat = gr.Chatbot(
                    label="Chat with Emu3.5",
                    height=550,
                    avatar_images=("assets/user.png", "assets/emu.png"),
                    elem_classes="chatbot"
                )
                state = gr.State([])

                text = gr.Textbox(
                    label="💬 Prompt",
                    placeholder="Enter your prompt here...",
                    lines=2
                )
                files = gr.Files(label="📷 Upload image(s)", file_count="multiple", type="filepath")

                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    stop = gr.Button("Stop", variant="secondary")
                    clear = gr.Button("Clear", variant="secondary")

        # 绑定回调
        init_btn.click(startup_initialize, [cfg, save_dir, device], status)
        send.click(on_submit, [text, files, mode, state], [chat, text, files, state])
        stop.click(on_stop, outputs=[status])
        clear.click(clear_chat, outputs=[chat, state])

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/config.py")
    parser.add_argument("--save_dir", type=str, default="./outputs")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    print(startup_initialize(args.cfg, args.save_dir, args.device))
    ui = build_ui()
    ui.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()