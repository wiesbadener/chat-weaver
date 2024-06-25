import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from the .env file (if present)
load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def generate_question(system_prompt, user_example, assistant_example, chat_input, temperature, max_tokens):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if user_example:
        messages.append({"role": "user", "content": user_example})
    if assistant_example:
        messages.append({"role": "assistant", "content": assistant_example})
    if chat_input:
        messages.append({"role": "user", "content": chat_input})
    
    response = client.chat.completions.create(model='gpt-3.5-turbo',
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
        stop=None
    )

    return response.choices[0].message.content

def on_click(system_prompt, user_example, assistant_example, chat_input, temperature, max_tokens):
    response = generate_question(
        system_prompt, 
        user_example, 
        assistant_example, 
        chat_input, 
        temperature,
        max_tokens
    )
    return [(chat_input, response)]

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, label="Temperature")
            max_tokens = gr.Slider(minimum=1, maximum=2048, value=150, label="Max Tokens")
        with gr.Column(scale=4):
            with gr.Accordion("System prompt and single-shot example", open=False):
                system_prompt = gr.Textbox(lines=3, placeholder="Enter system prompt here...", label="System Prompt")
                user_example = gr.Textbox(lines=3, placeholder="Enter example user prompt here...", label="User Example")
                assistant_example = gr.Textbox(lines=3, placeholder="Enter example assistant output here...", label="Assistant Example")

            chatbot = gr.Chatbot()
            user_prompt = gr.Textbox(lines=2, placeholder="Enter your text here...", label="User prompt")
            
            generate_button = gr.Button("Generate Response")
            generate_button.click(on_click, inputs=[system_prompt, user_example, assistant_example, user_prompt, temperature, max_tokens], outputs=[chatbot])
    
demo.launch()
