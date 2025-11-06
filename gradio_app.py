# -*- coding: utf-8 -*-
import argparse
import gradio as gr
import tempfile
import os
import shutil
from model_runtime import ModelRuntime

_RUNTIME = ModelRuntime.instance()

CSS = """
.chatbot .message.user { background: #f0fff0 !important; }
.chatbot .message.assistant { background: #eef2ff !important; }
"""

def _dup_path(src: str) -> str:
    """复制一个全新文件，避免同一路径在多条消息里复用导致的渲染问题。"""
    _, ext = os.path.splitext(src)
    tmp = tempfile.NamedTemporaryFile(suffix=ext or ".png", delete=False)
    tmp.close()
    shutil.copyfile(src, tmp.name)
    return os.path.abspath(tmp.name)

def startup_initialize(cfg_path: str, save_dir: str, device_str: str | None = None):
    return _RUNTIME.initialize(cfg_path=cfg_path, save_dir=save_dir, device_str=device_str)

def on_submit(text, files, mode, history):
    # 1) 更新采样配置
    _RUNTIME.update_sampling_config(mode)

    # 2) 设置样本（文本+图片路径）
    sample = {"text": text, "images": [f.name for f in files] if files else []}
    _RUNTIME.encode_and_set_prompt(sample)

    # 3) 先把“用户消息”塞进 Chatbot（tuple 格式）
    if files:
        history.append(("user", text))  # 显示用户的文本
        history.append(("user", [f.name for f in files]))  # 显示上传的文件
    else:
        history.append(("user", text))  # 如果没有文件，则只显示文本
    yield history, "", None, history  # 清空输入框/文件

    # 4) 占位一条 assistant 消息，后续 streaming 不断覆盖这条
    assistant_acc = ""
    history.append(("assistant", assistant_acc))
    yield history, "", None, history

    # 5) 消费流式事件
    for ev in _RUNTIME.stream_events(text_chunk_tokens=64):
        if ev["type"] == "text":
            # 如果是文本，继续拼接之前的文本
            assistant_acc += ev["text"]
            history[-1] = ("assistant", assistant_acc)
            yield history, "", None, history

        elif ev["type"] == "image":
            # 如果是图片，清空之前的文本，并开始新的文本生成
            for ip in ev.get("paths", []):
                echoed = _dup_path(ip)  # 复制图片路径以避免重复
                history.append(("assistant", [echoed]))  # 将图片路径作为内容添加到历史记录中
                yield history, gr.update(value=None), gr.update(value=None), history
            
            assistant_acc = ""  # 清空文本
            history.append(("assistant", assistant_acc))


def clear_chat():
    # 清空后端状态 + 返回两个输出：chat, state
    _RUNTIME.history.clear()
    return [], []

def on_stop():
    # 只发停止信号；前端通过绑定到 status 文本组件，立刻给出反馈
    _RUNTIME.request_stop()
    return "🛑 已发送停止信号（本轮生成将尽快结束显示）"

def build_ui():
    with gr.Blocks(css=CSS) as demo:
        gr.Markdown("# 🦄 Emu 3.5 Streaming Demo")

        with gr.Row():
            with gr.Column(scale=2):
                cfg = gr.Textbox(label="🧩 Config Path", value="configs/config.py")
                save_dir = gr.Textbox(label="📁 Output Dir", value="./outputs")
                device = gr.Textbox(label="⚙️ Device", value="")
                mode = gr.Dropdown(
                    label="Generation Mode",
                    choices=["default", "howto", "story", "t2i", "x2i"],
                    value="default"
                )
                init_btn = gr.Button("🚀 Load Model", variant="primary")
                status = gr.Markdown("")  # ⬅️ 停止按钮把文案写到这里

            with gr.Column(scale=6):
                # ⚠️ 使用默认的 tuple 模式（不要设置 type="messages"）
                chat = gr.Chatbot(label="Chat", height=540, elem_classes="chatbot")
                state = gr.State([])

                text = gr.Textbox(label="💬 Prompt", placeholder="Enter your prompt...", lines=2)
                files = gr.Files(label="📷 Upload image(s)", file_count="multiple", type="filepath")

                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    stop = gr.Button("Stop")
                    clear = gr.Button("Clear")

        # 绑定
        init_btn.click(startup_initialize, [cfg, save_dir, device], [status])

        # send -> (chat, text, files, state) 四个输出（对应 on_submit 的 yield）
        send.click(on_submit, [text, files, mode, state], [chat, text, files, state])

        # stop -> 输出 status（所以 on_stop 必须 return 字符串）
        stop.click(on_stop, outputs=[status])

        # clear -> 输出 chat 和 state 两个对象（所以 clear_chat 必须 return 两个值）
        clear.click(clear_chat, outputs=[chat, state])

    return demo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/config.py")
    parser.add_argument("--save_dir", type=str, default="./outputs")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    print(startup_initialize(args.cfg, args.save_dir, args.device))
    ui = build_ui()
    ui.queue()  # 建议开启队列，体验更稳定
    ui.launch(
        server_name=args.host,
        server_port=args.port,
        # show_error=True,
        # prevent_thread_lock=True,
        # allowed_paths=["."],        # 允许访问生成图片目录
        # enable_monitoring=False,    # ✅ 禁用 startup-events
    )

if __name__ == "__main__":
    main()