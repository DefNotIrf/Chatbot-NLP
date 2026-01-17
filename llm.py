import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------
# OpenAI 
# ---------------------------------------------------------
class OpenAILLM:
    def __init__(self):
        from openai import OpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Error: OPENAI_API_KEY not found.")
            
        self.client = OpenAI(api_key=api_key)

    def generate(self, system_prompt, user_query, context, history=[]):
        # 1. Start with System Prompt
        messages = [{"role": "system", "content": system_prompt}]
        
        # 2. Add Chat History (Previous Q&A)
        # We limit to the last 4 turns to save tokens
        messages.extend(history[-4:]) 
        
        # 3. Add Current Context & Query
        # We make the current context explicit so the LLM knows what we are looking at NOW
        current_prompt = f"Relevant Menu Info:\n{context}\n\nUser Question: {user_query}"
        messages.append({"role": "user", "content": current_prompt})
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3
        )
        return response.choices[0].message.content

# ---------------------------------------------------------
# Local Qwen
# ---------------------------------------------------------
class LocalQwenLLM:
    def __init__(self, model_path):
        from llama_cpp import Llama
        print("Loading Local Qwen Model...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,
            verbose=False
        )

    def generate(self, system_prompt, user_query, context):
        # ChatML format for Qwen
        prompt = (
            f"<|im_start|>system\n{system_prompt}\nContext:\n{context}<|im_end|>\n"
            f"<|im_start|>user\n{user_query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        output = self.llm(
            prompt, 
            max_tokens=150, 
            stop=["<|im_end|>"], 
            temperature=0.3
        )
        return output["choices"][0]["text"].strip()