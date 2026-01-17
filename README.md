# KAYNO: University Cafeteria Assistant

**KAYNO** (pronounced 'K-No') is an AI-powered cafeteria strategist designed to help students navigate meal options. The name is inspired by the local slang "Make Mano?" (Eat where?).

## 🚀 Features

* **Multimodal Input**: Supports both keyboard/text input and microphone/voice input.
* **Semantic RAG**: Uses a Retrieval-Augmented Generation engine to find meals based on context rather than just keywords.
* **Voice Optimized**: Automatically converts currency symbols like "RM" to "ringgit" for natural-sounding speech synthesis.
* **Local & Cloud Support**: Can run using OpenAI's GPT-4o-mini or locally on devices like a Raspberry Pi using Qwen.

---

## 🛠️ Project Structure

* **audio.py**: Manages the voice interface using Faster-Whisper for STT and Edge-TTS for speech.
* **llm.py**: Contains the logic for interacting with OpenAI and local Llama-CPP models.
* **rag.py**: Implements the search engine using FAISS and Sentence-Transformers.
* **siddiq_menu.csv**: The database containing stall names, meals, and prices.
* **laptop_voice.py / laptop_text.py**: Main entry points for laptop use.
* **pi_local.py**: Entry point for local execution on a Raspberry Pi.

---

## 🔧 Installation

1. **Clone the repository** and navigate to the project folder.
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. Setup Environment: Create a .env file and add your OpenAI API key

---

## 🤖 Usage
Laptop (Cloud-based)
To use the voice assistant:
```bash
python laptop_voice.py
```

To use the text-only assistant:
```bash
python laptop_text.py
```

Raspberry Pi (Local-based)
Ensure your GGUF model is placed in the models/ folder, then run:
```bash
python pi_local.py
```
---
