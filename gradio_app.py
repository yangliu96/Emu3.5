import gradio as gr
from model_runtime import ModelRuntime

_RUNTIME = ModelRuntime.instance()

def on_submit(text, files, mode, history):
    _RUNTIME.update_sampling_config(mode)

    file_paths = [f.name for f in files] if files else []
    sample = {"text": text, "images": file_paths}
    _RUNTIME.encode_and_set_prompt(sample)

    for ev in _RUNTIME.stream_events(max_rounds=64, text_chunk_tokens=64):
        if ev["type"] == "text":
            history.append(("assistant", ev["text"]))
            yield history, "", None, history
        elif ev["type"] == "image":
            history.append(("assistant", ev["paths"]))
            yield history, "", None, history

def on_stop():
    _RUNTIME.request_stop()

def on_clear():
    _RUNTIME.clear_history()
    return [], "", None, []


with gr.Blocks(title="Emu3.5 Multi-image") as demo:
    chatbot = gr.Chatbot(height=560)

    with gr.Row():
        tb = gr.Textbox(label="Text", lines=2)
        mode_dd = gr.Dropdown(["default", "howto", "story", "t2i", "x2i"], value="default", label="Mode")
        files = gr.Files(label="Images", file_types=["image"], file_count="multiple")

    with gr.Row():
        send = gr.Button("发送", variant="primary")
        stop = gr.Button("停止")
        clear = gr.Button("清空")

    state_history = gr.State([])

    send.click(on_submit, inputs=[tb, files, mode_dd, state_history],
               outputs=[chatbot, tb, files, state_history])
    tb.submit(on_submit, inputs=[tb, files, mode_dd, state_history],
              outputs=[chatbot, tb, files, state_history])

    stop.click(on_stop)
    clear.click(on_clear, outputs=[chatbot, tb, files, state_history])

demo.launch()