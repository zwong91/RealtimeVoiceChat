# Real-Time AI Voice Chat 🎤💬🧠🔊

*early preview - current state still contains lots of bugs and unsolved edge cases*

Implements a real-time, voice-based chat application where users can speak directly to an AI assistant and receive spoken responses, mimicking a natural conversation. It's based on a client-server low-latency communication architecture with WebSockets.

## Overview

The system captures microphone audio from a web client, streams it to a Python backend server, transcribes the audio to text in real-time, processes the text using a Large Language Model (LLM), synthesizes the AI's text response back into audio, and streams the audio back to the client for playback. It features interruption handling, turn detection, and displays partial transcriptions/responses.

## Features

*   **Real-Time Voice Interaction:** Speak naturally and get spoken responses from the AI.
*   **Client-Server Architecture:** Web-based client connects to a powerful Python backend via WebSockets.
*   **Low Latency:** Optimized for minimal delay using audio chunk streaming.
*   **Real-Time Transcription:** Uses `RealtimeSTT` for fast and accurate speech-to-text conversion.
*   **Turn Detection:** Employs a model (`turndetect.py`) to dynamically adjust silence thresholds for natural conversation flow.
*   **LLM Integration:** Connects to LLMs (configurable, supports Ollama and potentially OpenAI, `llm_module.py`).
*   **Real-Time Text-to-Speech (TTS):** Uses `RealtimeTTS` with various engine options (Kokoro, Coqui, Orpheus) to generate spoken audio (`audio_module.py`).
*   **Partial & Final Responses:** Displays user transcriptions and AI responses as they are generated.
*   **Interruption Handling:** Allows the user to interrupt the AI's response by speaking.
*   **Web-Based UI:** Simple and clean chat interface using HTML, CSS, and JavaScript (`static/`).
*   **Audio Worklets:** Efficient client-side audio processing for capture and playback.

## Technology Stack

*   **Backend:** Python 3.x, FastAPI
*   **Frontend:** HTML, CSS, JavaScript (Vanilla JS, Web Audio API with AudioWorklets)
*   **Communication:** WebSockets
*   **Core AI/ML Libraries:**
    *   `RealtimeSTT` (Speech-to-Text)
    *   `RealtimeTTS` (Text-to-Speech)
    *   `transformers` (For turn detection model, LLM tokenization)
    *   `torch` / `torchaudio` (Required by STT/TTS/Transformers)
    *   `ollama` / `openai` (LLM Interaction)
*   **Audio Processing:** `numpy`, `scipy`
*   **Environment:** Virtual Environment (`venv`)

## Setup and Installation

**Prerequisites:**

*   Python 3.9 or higher recommended.
*   Windows recommended.
*   A CUDA-enabled **STRONG** GPU is highly recommended for faster STT/TTS performance (especially for Coqui TTS and larger Whisper models). The installation script assumes CUDA 12.4 (`cu124`). Adjust if necessary.
*   **(Optional) Ollama:** If using the Ollama backend for the LLM, ensure it is installed and running. Pull the desired model (e.g., `ollama pull hf.co/bartowski/huihui-ai_Mistral-Small-24B-Instruct-2501-abliterated-GGUF:Q4_K_M`), set in handlerequests.py as MODEL parameter.
*   **(Optional) OpenAI API Key:** If using the OpenAI backend, set the `OPENAI_API_KEY` environment variable or place it in a `.env` file.

**Installation Steps:**

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Run the installation script (Windows):**
    This script automates creating a virtual environment, upgrading pip, installing PyTorch with CUDA support, and installing all other dependencies from `requirements.txt`.
    ```batch
    install.bat
    ```
    *(Note: This will open a new command prompt window within the activated virtual environment.)*

3.  **Manual Installation (Linux/macOS or if `.bat` fails):**
    ```bash
    # Create virtual environment
    python -m venv venv

    # Activate virtual environment
    # Linux/macOS:
    source venv/bin/activate
    # Windows:
    .\venv\Scripts\activate

    # Upgrade pip
    python -m pip install --upgrade pip

    # Navigate to the code directory
    cd code

    # Install PyTorch with CUDA (adjust index-url for your CUDA version or CPU)
    # Example for CUDA 12.4:
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
    # Example for CPU only:
    # pip install torch torchaudio

    # Install other requirements
    pip install -r requirements.txt
    ```

## Running the Application

1.  **Ensure your virtual environment is activated.**
    ```bash
    # If not already active:
    # Linux/macOS: source ../venv/bin/activate
    # Windows: ..\venv\Scripts\activate
    ```

2.  **Make sure you are in the `code` directory.**

3.  **Start the FastAPI server:**
    ```bash
    python server.py
    ```

4.  **Access the client:**
    Open your web browser and navigate to `http://localhost:8000` (or `http://<your-server-ip>:8000` if running on a different machine).

5.  **Grant microphone permissions** when prompted by the browser.

6.  **Click the "Start" button** to begin the voice chat. Click "Stop" to end the session. Use "Reset" to clear the conversation history on both client and server.

## Configuration

Several aspects of the application can be configured by modifying the Python source files:

*   **TTS Engine (`server.py`, `audio_module.py`):**  
  Change `START_ENGINE` in `server.py` to "coqui", "kokoro", or "orpheus". Configure engine-specific settings (voice, speed, etc.) within `AudioProcessor.__init__` in `audio_module.py`. 
  When you choose CoquiEngine, I recommend running DeepSpeed. On Linux, you can just `pip install deepspeed`, but on Windows you'll need to build it yourself. A handy tool for that is [deepspeedpatcher](https://github.com/erew123/deepspeedpatcher). When you compile, enable the “CUTLESS_OPS,” “SPARSE_ATTN,” and “INFERENCE_CORE_OPS” options. You'll have to install a few prerequisites first (check the repo's README), but the performance boost is well worth the extra setup.
*   **LLM Model & Backend (`llm_module.py`):**
    *   Set the desired `LLM_START_MODEL` (e.g., Ollama model name or HF path) and `LLM_START_PROVIDER`  in `server.py`.
    *   Adjust system prompt in `system_prompt.txt`
*   **STT Settings (`transcribe.py`):** Modify `DEFAULT_RECORDER_CONFIG` settings at the top of the file to change Whisper model size, language, sensitivities, silence durations, etc.
*   **Turn Detection (`turndetect.py`):** Adjust pause duration constants in `update_settings` method.
* **SSL (`server.py`):**  
  Set `USE_SSL = True` and provide certificate/key files if HTTPS is required.  
  To generate local certificates on Windows:
  1. Run your command prompt as administrator.
  2. Install mkcert with Chocolatey:
     ```bat
     choco install mkcert
     ```
  3. Install the local CA:
     ```bat
     mkcert -install
     ```
  4. Create certificates for localhost and your local IP:
     ```bat
     mkcert 127.0.0.1 192.168.178.123
     
## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs, feature requests, or improvements.

## License

My own Codebase is MIT.
Please respect the license systems of TTS providers.
