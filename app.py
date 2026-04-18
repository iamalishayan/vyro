import gradio as gr
from inference import run

def chat(message, history):
    # Convert Gradio's history format to our format
    formatted_history = []
    for user_msg, bot_msg in history:
        formatted_history.append({"role": "user", "content": user_msg})
        if bot_msg:
            formatted_history.append({"role": "assistant", "content": bot_msg})

    response = run(message, formatted_history)
    history.append((message, response))
    return "", history

with gr.Blocks(title="Pocket-Agent") as demo:
    gr.Markdown("# 🤖 Pocket-Agent")
    gr.Markdown("Tools: `weather` · `calendar` · `convert` · `currency` · `sql`")

    chatbot = gr.Chatbot(height=450)
    msg = gr.Textbox(placeholder="Ask me anything...", show_label=False)
    clear = gr.ClearButton([msg, chatbot])

    msg.submit(chat, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)
