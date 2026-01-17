from src.rag import RAGEngine
from src.llm import OpenAILLM
from src.audio import TextIO

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


PERSONALITY:
- Be helpful and friendly.
- If asked about your name, you are KAYNO, which comes from the local slang 'Make Mano?' (Eat where?).
"""

def main():
    rag = RAGEngine(MENU_PATH)
    llm = OpenAILLM()
    io = TextIO()
    chat_history = []

    welcome_msg = "Hello! I am KAYNO, your meal assistant. How can I help you eat today?"
    io.output(welcome_msg)

    while True:
        user_query = io.get_input()
        if not user_query: continue
        if any(word in user_query.lower() for word in ["thank", "thanks", "exit"]):
                io.output("You are welcome! Enjoy your meal. Jumpa lagi.")
                break

        context_data = rag.search(user_query)

        response = llm.generate(SYSTEM_PROMPT, user_query, context_data, chat_history)

        chat_history.append({"role": "user", "content": user_query})
        chat_history.append({"role": "assistant", "content": response})

        io.output(response)

if __name__ == "__main__":

    main()
