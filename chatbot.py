from audio.record_audio import record_audio
from stt.whisper_stt import WhisperSTT
from tts.tts_engine import TTSEngine

from dialogue.dialogue_manager import DialogueManager
from llm.question_generator import QuestionGenerator

from data.menu_loader import load_menu_data
from data.retriever import retrieve_menu
from data.context_builder import build_menu_context
from llm.llm_engine import LLMEngine

# Initialize
tts = TTSEngine()
stt = WhisperSTT()
dialogue = DialogueManager()
question_llm = QuestionGenerator()

menu_df = load_menu_data()

# -----------------------
# Dialogue loop
# -----------------------
while dialogue.missing_slots():

    missing_slot = dialogue.missing_slots()[0]

    question = question_llm.ask_question(missing_slot)
    tts.speak(question)

    record_audio(duration=5)
    user_input = stt.transcribe("audio/user_input.wav")

    dialogue.update_slot(user_input)

# -----------------------
# RAG
# -----------------------
filtered_df = retrieve_menu(
    menu_df,
    dialogue.slots.get("stall_name", "")
)

menu_context = build_menu_context(filtered_df)

llm = LLMEngine(menu_context)

final_response = llm.generate_response(
    "Suggest a suitable meal based on the preferences."
)

tts.speak(final_response)
