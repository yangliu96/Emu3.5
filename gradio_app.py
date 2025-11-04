# -*- coding: utf-8 -*-
import gradio as gr
from model_runtime import ModelRuntime


# -------------------- Runtime 预加载 --------------------
_RUNTIME = ModelRuntime.instance()

def startup_initialize(cfg_path: str, save_dir: str, device_str: str = None):
    """启动时预加载模型"""
    return _RUNTIME.initialize(cfg_path=cfg_path, save_dir=save_dir, device_str=device_str)


# -------------------- Gradio 回调 --------------------
def on_submit(text, files, mode, history):
    """发送按钮事件"""
    _RUNTIME.update_sampling_config(mode)

    file_paths = [f.name for f in files] if files else []
    sample = {"text": text, "images": file_paths}
    _RUNTIME.encode_and_set_prompt(sample)

    for ev in _RUNTIME.stream_events(text_chunk_tokens=64):
        if ev["type"] == "text":
            history.append(("assistant", ev["text"]))
            yield history, "", None, history
        elif ev["type"] == "image":
            history.append(("assistant", ev["paths"]))
            yield history, "", None, history


def on_stop():
    """停止（下一个 chunk 完成即可停止）"""
    _RUNTIME.request_stop()


def on_clear():
    """清空聊天记录 / 清空内部上下文"""
    _RUNTIME.clear_history()
    return [], "", None, []


# -------------------- UI --------------------
with gr.Blocks(title="Emu3.5 Multi-image") as demo:
    chatbot = gr.Chatbot(height=560)

    with gr.Row():
        tb = gr.Textbox(label="Text", lines=2, placeholder="输入提示词...")
        mode_dd = gr.Dropdown(
            ["default", "howto", "story", "t2i", "x2i"],
            value="default",
            label="Mode"
        )
        files = gr.Files(
            label="Images",
            file_types=["image"],
            file_count="multiple"
        )

    with gr.Row():
        send = gr.Button("发送", variant="primary")
        stop = gr.Button("停止")
        clear = gr.Button("清空")

    state_history = gr.State([])

    send.click(
        on_submit,
        inputs=[tb, files, mode_dd, state_history],
        outputs=[chatbot, tb, files, state_history]
    )
    tb.submit(
        on_submit,
        inputs=[tb, files, mode_dd, state_history],
        outputs=[chatbot, tb, files, state_history]
    )
    stop.click(on_stop)
    clear.click(on_clear, outputs=[chatbot, tb, files, state_history])


# =====================================================================
# ✅ CLI entry：支持启动参数：--port --host --cfg --save_dir --device
# =====================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument("--port", type=int, default=None, help="Port for Gradio server")
    parser.add_argument("--host", type=str, default=None, help="Host for Gradio server (e.g. 0.0.0.0)")
    parser.add_argument("--cfg", type=str, default=None, help="Config path for model init (overrides UI)")
    parser.add_argument("--save_dir", type=str, default=None, help="Output directory for generated images"
                                                                   )
    parser.add_argument("--device", type=str, default=None, help="Device string (e.g., cuda:0 or cpu)")
    args, _ = parser.parse_known_args()


    # 默认加载 config/app_config.py
    cfg_path = args.cfg or "configs/config.py"
    save_dir = args.save_dir or "./outputs"
    device_str = args.device or None

    print(startup_initialize(cfg_path=cfg_path, save_dir=save_dir, device_str=device_str))

    launch_kwargs = {}
    if args.port is not None:
        launch_kwargs["server_port"] = args.port
    if args.host is not None:
        launch_kwargs["server_name"] = args.host

    demo.launch(**launch_kwargs)