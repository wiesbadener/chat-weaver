# SPDX-FileCopyrightText: 2024-present wiesbadener
#
# SPDX-License-Identifier: Apache-2.0

import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from the .env file (if present)
load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

SYSTEM_PROMPT = "Read the provided text and generate a question based on the context."

USER_EXAMPLE = "Today I went fishing. I caught a big salmon. After that, I went on a hike in the mountains. The path was not easy, but the view was worth it. I followed the yellow signs to reach the summit. I saw a beautiful sunset from the top. It took me a while to get back down, but it was a great day."

ASSISTANT_EXAMPLE = "How did you navigate to reach the mountain top?"

def get_user_prompt() -> str:
    # Example user prompt. Add some logic that yields text to generate questions about it.
    EXAMPLE_USER_PROMPT = """In a small village, a curious boy named Sam discovered a hidden path in the forest. Following it, he found a sparkling pond where an old woman sat weaving a garland of blue flowers. She told him, "This is the Pond of Wishes. Throw a stone and make a true wish." Sam picked up a stone, closed his eyes, made a wish, and threw it into the pond. Weeks later, the village woke up to a beautiful garden of blue flowers around the pond, with Sam smiling in the center. His wish had come true."""
    return EXAMPLE_USER_PROMPT

def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)

def generate_questions(
        system_prompt: str, 
        user_example: str, 
        assistant_example: str, 
        chat_input: str, 
        temperature: float, 
        max_tokens: int, 
        model_name: str
        ) -> str:
    """
    Generates a question using the OpenAI Chat API.

    Args:
        system_prompt (str): The system prompt to set the context for the conversation.
        user_example (str): An example of a user message.
        assistant_example (str): An example of an assistant message.
        chat_input (str): The user's input message.
        temperature (float): Controls the randomness of the output. Higher values make the output more random.
        max_tokens (int): The maximum number of tokens to generate in the response.
        model_name (str): The name of the model to use for generating the response.
        chat_history (list): A list of tuples containing the chat history, where each tuple contains a user message and an assistant message.

    Returns:
        str: The generated question.
        list: List of generated questions.

    """
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if user_example:
        messages.append({"role": "user", "content": user_example})
    if assistant_example:
        messages.append({"role": "assistant", "content": assistant_example})
    if chat_input:
        messages.append({"role": "user", "content": chat_input})
    
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        n=2,
        stop=None
    )

    return [choice.message.content for choice in response.choices]

def on_click(
        system_prompt: str, 
        user_example: str, 
        assistant_example: str, 
        chat_input: str, 
        temperature: float, 
        max_tokens: int, 
        model_name: str, 
        chat_history: list
        ) -> tuple[str, list]:
    """
    Generates a response based on the given inputs and updates the chat history.

    Args:
        system_prompt (str): The system prompt for generating the response.
        user_example (str): An example of user input.
        assistant_example (str): An example of assistant input.
        chat_input (str): The user's input for the current chat.
        temperature (float): The temperature parameter for response generation.
        max_tokens (int): The maximum number of tokens in the generated response.
        model_name (str): The name of the model used for response generation.
        chat_history (list): The chat history containing previous user and assistant inputs.

    Returns:
        tuple[str, list]: A tuple containing a empty string to reset user prompt textbox and the updated chat history.
    """
    
    response_list = generate_questions(
        system_prompt, 
        user_example, 
        assistant_example, 
        chat_input, 
        temperature,
        max_tokens,
        model_name,
        chat_history
    )
    empty_user_inputt = ""
    chat_history.append([chat_input, response_list])
    return empty_user_inputt, chat_history

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            temperature = gr.Slider(minimum=0.0, maximum=2.0, value=0.7, label="Temperature")
            max_tokens = gr.Slider(minimum=1, maximum=2048, value=20, label="Max Tokens")
            model_name = gr.Dropdown(choices=["gpt-3.5-turbo", 'Other'], value="gpt-3.5-turbo", label="Model Name")
        with gr.Column(scale=4):
            with gr.Accordion("System prompt and single-shot example", open=False):
                system_prompt = gr.Textbox(value=SYSTEM_PROMPT, placeholder="Enter system prompt here...", label="System Prompt")
                user_example = gr.Textbox(value=USER_EXAMPLE, lines=3, placeholder="Enter example user prompt here...", label="User Example")
                assistant_example = gr.Textbox(value=ASSISTANT_EXAMPLE, placeholder="Enter example assistant output here...", label="Assistant Example")

            user_prompt = gr.Textbox(value=get_user_prompt, placeholder="Enter your text here...", label="User prompt")
            
            @gr.render(inputs=[system_prompt, user_example, assistant_example, user_prompt, temperature, max_tokens, model_name], triggers=[user_prompt.submit])
            def show_split(system_prompt, user_example, assistant_example, user_prompt, temperature, max_tokens, model_name):
                if len(user_prompt) == 0:
                    gr.Markdown("## No Input Provided")
                else:
                    response_list = generate_questions(
                        system_prompt, 
                        user_example, 
                        assistant_example, 
                        user_prompt, 
                        temperature,
                        max_tokens,
                        model_name
                    )

                gr.CheckboxGroup(response_list, label="Questions", info="List of questions generated by the model.")
                # Note that all event listeners that use Components created inside a render function must also be defined inside that render function. 
            
demo.launch()
