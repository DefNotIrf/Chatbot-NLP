from modules.rag import RAGEngine
from modules.llm import LocalQwenLLM
from modules.audio import VoiceIO


MENU_PATH = "\\data\\siddiq_menu.csv"
MODEL_PATH = "models/qwen2.5-3b-instruct-q4_k_m.gguf"
SYSTEM_PROMPT = """
You are a university cafeteria assistant.

You MUST follow this reasoning process:
1. Identify what the user is asking (price, recommendation, time, tags).
2. Look ONLY at the provided menu information.
3. Compare meals logically if needed.
4. Answer clearly in one or two sentences.

Rules:
- DO NOT invent meals.
- DO NOT guess prices.
- If information is missing, say you don't know.
"""

def main():
    # 1. Initialize Modules
    rag = RAGEngine(MENU_PATH)
    llm = LocalQwenLLM(MODEL_PATH)
    io = VoiceIO()

    io.output("System ready. I am listening.")

    # 2. Main Loop
    while True:
        try:
            user_query = io.get_input()
            if not user_query: continue
            if "exit" in user_query.lower(): break

            # Retrieve & Generate
            context_data = rag.search(user_query)
            response = llm.generate(SYSTEM_PROMPT, user_query, context_data)

            io.output(response)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()