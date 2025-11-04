# app.py
import gradio as gr
import os
from typing import List, Any
from PIL import Image
import re
import argparse

from model_runtime import ModelRuntime

# ---------------- 运行时封装（热重载/中断/裁剪） ----------------
_RUNTIME = ModelRuntime.instance()

def startup_initialize(cfg_path: str = "configs/config.py",
                       save_dir: str = "./outputs",
                       device_str: str = None) -> str:
    """
    启动时初始化模型，仅加载一次。
    """
    return _RUNTIME.initialize(cfg_path=cfg_path, save_dir=save_dir, device_str=device_str)

def runtime_reload(cfg_path: str, save_dir: str, device_str: str) -> str:
    """
    热重载模型，重新加载配置和权重。
    """
    return _RUNTIME.initialize(cfg_path=cfg_path, save_dir=save_dir, device_str=device_str, force_reload=True)

def runtime_clear_history() -> str:
    """
    清空对话历史。
    """
    try:
        _RUNTIME.clear_history()
        return "🧹 已清空对话历史。"
    except Exception as e:
        return f"清空历史失败：{e}"

def runtime_request_stop() -> str:
    """
    请求停止当前生成。
    """
    _RUNTIME.request_stop()
    return "⏹️ 已请求停止当前生成。"

def _split_sentences_cn_en(s: str) -> List[str]:
    """
    按中英文标点切分长文本。
    """
    if not s:
        return []
    parts = re.split(r'([。！？；!?;])', s)
    return [p.strip() for p in parts if p.strip()]

def _chunk_text_cn_en(s: str, max_len: int = 80) -> List[str]:
    """
    将长文本分块，避免前端显示过长的文本。
    """
    sentences = _split_sentences_cn_en(s)
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_len:
            current_chunk += sentence
        else:
            chunks.append(current_chunk)
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def on_submit(text: str, files: List[Any], mode: str, history: List[gr.ChatMessage]):
    """
    提交用户输入，触发生成。
    """
    text = (text or "").strip()
    file_paths = [f.name for f in files] if files else []

    # 切换 mode 时更新采样参数
    _RUNTIME.update_sampling_config(mode)

    # 左侧：用户输入的文本和图片
    if text:
        history.append(gr.ChatMessage(role="user", content=text))
        yield history, gr.update(value=None), gr.update(value=None), history

    for file_path in file_paths:
        try:
            img = Image.open(file_path).convert("RGB")
            history.append(gr.ChatMessage(role="user", content=[file_path]))
            yield history, gr.update(value=None), gr.update(value=None), history
        except Exception:
            pass

    # 生成准备
    _RUNTIME.reset_stop()
    raw_sample = {"text": text, "images": file_paths, "mode": mode}
    _RUNTIME.encode_and_set_prompt(raw_sample)

    # 流式生成
    for ev in _RUNTIME.stream_events(max_rounds=64, text_chunk_tokens=64):
        if ev["type"] == "text":
            chunks = _chunk_text_cn_en(ev["text"], max_len=80)
            for chunk in chunks:
                history.append(gr.ChatMessage(role="assistant", content=chunk))
                yield history, gr.update(value=None), gr.update(value=None), history
        elif ev["type"] == "image":
            history.append(gr.ChatMessage(role="assistant", content=ev["paths"]))
            yield history, gr.update(value=None), gr.update(value=None), history

def on_clear():
    """
    清空对话历史。
    """
    _RUNTIME.reset_stop()
    _RUNTIME.clear_history()
    return [], None, None, []

# ---------------- UI ----------------

with gr.Blocks(title="Model Text + Multi-Image") as demo:
    gr.Markdown("### 输入文本与多图；右侧按**生成步骤**依次输出：支持任务切换、Stop、清空、热重载。")

    chatbot = gr.Chatbot(type="messages", height=560, label="Conversation")

    with gr.Row():
        tb = gr.Textbox(label="Text", placeholder="输入文本...", lines=2, scale=5)
        mode_dd = gr.Dropdown(
            label="任务类型",
            choices=["default", "howto", "story", "t2i", "x2i"],
            value="default",
            scale=2,
        )
        files = gr.Files(label="上传图片", file_types=["image"], file_count="multiple", scale=5)

    with gr.Row():
        send = gr.Button("发送", variant="primary")
        stop = gr.Button("停止")
        clear = gr.Button("清空")

    with gr.Accordion("高级设置", open=False):
        cfg_path_tb = gr.Textbox(label="配置文件路径", value="configs/config.py", scale=4)
        save_dir_tb = gr.Textbox(label="输出目录", value="./outputs", scale=3)
        device_tb = gr.Textbox(label="设备", value="cuda:0", scale=2)
        reload_btn = gr.Button("重新加载模型", variant="secondary", scale=1)

    state_history = gr.State([])  # 历史记录
    ready_msg = gr.Markdown()
    demo.load(lambda: startup_initialize(), outputs=ready_msg)

    tb.submit(on_submit, inputs=[tb, files, mode_dd, state_history], outputs=[chatbot, tb, files, state_history])
    send.click(on_submit, inputs=[tb, files, mode_dd, state_history], outputs=[chatbot, tb, files, state_history])
    stop.click(lambda: runtime_request_stop(), outputs=[])
    clear.click(on_clear, outputs=[chatbot, tb, files, state_history])
    reload_btn.click(lambda cfg, sd, dev: runtime_reload(cfg, sd, dev),
                     inputs=[cfg_path_tb, save_dir_tb, device_tb],
                     outputs=ready_msg)

# ---------------- 命令行参数支持 ----------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--port", type=int, default=None, help="Port for Gradio server")
    parser.add_argument("--host", type=str, default=None, help="Host for Gradio server (e.g. 0.0.0.0)")
    parser.add_argument("--cfg", type=str, default=None, help="Config path for model init (overrides UI)")
    parser.add_argument("--save_dir", type=str, default=None, help="Output directory for generations")
    parser.add_argument("--device", type=str, default=None, help="Device string, e.g. cuda:0 or cpu")
    args, _ = parser.parse_known_args()

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