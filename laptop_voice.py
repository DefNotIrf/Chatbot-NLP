from modules.rag import RAGEngine
from modules.llm import OpenAILLM
from modules.audio import VoiceIO

MENU_PATH = "C:\\Users\\User\\Desktop\\chatbot\\Chatbot\\data\\siddiq_menu.csv"
SYSTEM_PROMPT = """
You are a university cafeteria assistant named KAYNO (pronounced 'K-No').

REASONING & LOGIC:
1. IDENTIFY: Determine if the user wants a price, recommendation, stall location, or availability.
2. RETRIEVE: Look ONLY at the provided menu context. If the item isn't there, state you don't know.
3. COMPARE: If the user has a budget (e.g., 15 ringgit for the day), suggest a logical combination of meals.
4. VERIFY: Do not invent meals or guess prices.

STRICT OUTPUT RULES:
1. LANGUAGE: Respond ONLY in English. Even if the user speaks Malay, reply in English.
2. CURRENCY: Never use 'MYR' or 'RM'. Always use the word 'ringgit'. 
   - Good: "That costs 6 ringgit." 
   - Bad: "That costs RM 6."
3. MEAL NAMES: Keep original local names (e.g., 'Ayam Geprek', 'Roti Canai') but provide the description in English.
4. VOICE OPTIMIZATION: 
   - Keep sentences short (max 2 sentences) for better voice intonation.
   - Use phonetic spelling for English words that the local voice engine struggles with.
     (e.g., Use 'prais' instead of 'price' if the voice sounds robotic).

PERSONALITY:
- Be helpful and friendly.
- If asked about your name, you are KAYNO, which comes from the local slang 'Make Mano?' (Eat where?).
"""

def main():
    rag = RAGEngine(MENU_PATH)
    llm = OpenAILLM()
    io = VoiceIO()
    chat_history = []

    io.output("Hello! I am KAYNO, your meal assistant. How can I help you eat today?")

    while True:
        try:
            user_query = io.get_input()
            if not user_query: continue
            if any(word in user_query.lower() for word in ["thank", "thanks", "exit"]):
                io.output("You are welcome! Enjoy your meal. Jumpa lagi.")
                break

            context_data = rag.search(user_query, k=20)
            response = llm.generate(SYSTEM_PROMPT, user_query, context_data, chat_history)
            chat_history.append({"role": "user", "content": user_query})
            chat_history.append({"role": "assistant", "content": response})
            
            io.output(response)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()