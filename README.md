# Real-Time AI Voice Chat 🎤💬🧠🔊

*early preview - current state still contains lots of bugs and unsolved edge cases*

https://github.com/user-attachments/assets/16cc29a7-bec2-4dd0-a056-d213db798d8f

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
*   **Containerization:** Docker, Docker Compose
*   **Core AI/ML Libraries:**
    *   `RealtimeSTT` (Speech-to-Text)
    *   `RealtimeTTS` (Text-to-Speech)
    *   `transformers` (For turn detection model, LLM tokenization)
    *   `torch` / `torchaudio` (Required by STT/TTS/Transformers)
    *   `ollama` / `openai` (LLM Interaction)
*   **Audio Processing:** `numpy`, `scipy`
*   **Environment:** Virtual Environment (`venv`) or Docker

## Setup and Installation

**Prerequisites:**

*   Python 3.9 or higher recommended (for manual/venv setup).
*   Windows recommended (for manual/venv setup).
*   Linux with Docker, Docker Compose, and NVIDIA Container Toolkit (for Docker setup).
*   A CUDA-enabled **STRONG** GPU is highly recommended for faster STT/TTS performance (especially for Coqui TTS and larger Whisper models). The installation script/Dockerfile assumes CUDA 12.1 (`cu121`). Adjust if necessary. The `docker-compose.yml` file is configured to pass GPU access to the service.
*   **(Optional) Ollama:** If using the Ollama backend *outside* of Docker, ensure it is installed and running. Pull the desired model manually. The Docker Compose setup handles Ollama installation and provides commands to pull models.
*   **(Optional) OpenAI API Key:** If using the OpenAI backend, set the `OPENAI_API_KEY` environment variable or place it in a `.env` file (or pass it as an environment variable to Docker, e.g., in your `docker-compose.yml` or via command line).

<details>

<summary><strong>Installation Steps</strong> (Click to expand/collapse)</summary>

**Clone the repository:**
```bash
git clone https://github.com/KoljaB/RealtimeVoiceChat.git
cd RealtimeVoiceChat
```

Choose one of the following installation methods:

**A) Docker Installation (Recommended for Linux/GPU):**

This method uses Docker Compose to build and manage the application and its dependencies, including Ollama if configured. Ensure you have Docker, Docker Compose v2+, and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed.

1.  **Build the Docker images:**
    *(This will take some time as it downloads base images, installs dependencies via the Dockerfile, and potentially pre-downloads ML models defined during the build)*
    ```bash
    docker compose build
    ```

2.  **Start the services (Application & Ollama):**
    *(Starts containers in detached mode. GPU access and port mapping are defined in `docker-compose.yml`)*
    ```bash
    docker compose up -d
    ```
    Wait a few moments for the services to initialize.

3.  **(Optional but Recommended) Pull the desired Ollama Model:**
    *(Execute this command *after* services are up to pull a specific model into the running Ollama container. The default model configured in `server.py` is `hf.co/bartowski/huihui-ai_Mistral-Small-24B-Instruct-2501-abliterated-GGUF:Q4_K_M`)*
    ```bash
    docker compose exec ollama ollama pull hf.co/bartowski/huihui-ai_Mistral-Small-24B-Instruct-2501-abliterated-GGUF:Q4_K_M
    ```

4.  **(Optional) Verify the model is pulled:**
    ```bash
    docker compose exec ollama ollama list
    ```

5.  **Stopping the services:**
    *(Stops and removes the containers defined in `docker-compose.yml`)*
    ```bash
    docker compose down
    ```

6.  **Restarting the services:**
    ```bash
    docker compose up -d
    ```

7.  **Viewing Logs / Debugging:**
    *   Watch logs for the main application:
        ```bash
        docker compose logs -f app
        ```
    *   Watch logs for the Ollama service:
        ```bash
        docker compose logs -f ollama
        ```
    *   If something goes wrong, capture the logs:
        ```bash
        docker compose logs app > app_logs.txt
        docker compose logs ollama > ollama_logs.txt
        ```
        *(Share these files when reporting issues)*

**B) Installation Script (Windows):**

This script automates creating a virtual environment, upgrading pip, installing PyTorch 2.5.1 with CUDA 12.1 support and a suitable Deepspeed wheel, and installing all other dependencies from `requirements.txt`.

1.  **Run the installation script:**
    ```batch
    install.bat
    ```
    *(Note: This will open a new command prompt window within the activated virtual environment.)*

**C) Manual Installation (Linux/macOS or if `.bat` fails):**

1.  **Create virtual environment:**
    ```bash
    python -m venv venv
    ```

2.  **Activate virtual environment:**
    *   Linux/macOS:
        ```bash
        source venv/bin/activate
        ```
    *   Windows:
        ```bash
        .\venv\Scripts\activate
        ```

3.  **Upgrade pip:**
    ```bash
    python -m pip install --upgrade pip
    ```

4.  **Navigate to the code directory:**
    ```bash
    cd code
    ```

5.  **Install PyTorch with CUDA (adjust index-url for your CUDA version or CPU):**
    *   Example for CUDA 12.1:
        ```bash
        pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 torchvision --index-url https://download.pytorch.org/whl/cu121
        ```
    *   Example for CPU only:
        ```bash
        # pip install torch torchaudio
        ```

6.  **Install other requirements:**
    ```bash
    pip install -r requirements.txt
    ```

</details>

## Running the Application

**If using Docker:**

1.  The application and required services (like Ollama) are started using the `docker compose up -d` command (Step A2 in Installation). See Step A7 for monitoring logs.

**If using Manual/Script Installation:**

1.  **Ensure your virtual environment is activated.**
    ```bash
    # If not already active:
    # Linux/macOS: source ../venv/bin/activate
    # Windows: ..\venv\Scripts\activate
    ```

2.  **Make sure you are in the `code` directory.**
    ```bash
    # If you ran install.bat, you might already be here.
    # Otherwise:
    cd code
    ```

3.  **Start the FastAPI server:**
    ```bash
    python server.py
    ```

**Accessing the Client (All Methods):**

1.  Open your web browser and navigate to `http://localhost:8000` (or `http://<your-server-ip>:8000` if accessing the server or Docker container remotely).

2.  **Grant microphone permissions** when prompted by the browser.

3.  **Click the "Start" button** to begin the voice chat. Click "Stop" to end the session. Use "Reset" to clear the conversation history on both client and server.

## Configuration

Several aspects of the application can be configured by modifying the Python source files (`code/` directory) *before* building the Docker image (`docker compose build`) or running `server.py`:

*   **TTS Engine (`server.py`, `audio_module.py`):**
    Change `START_ENGINE` in `server.py` to "coqui", "kokoro", or "orpheus". Configure engine-specific settings (voice, speed, etc.) within `AudioProcessor.__init__` in `audio_module.py`.
    When you choose CoquiEngine, DeepSpeed is recommended. The Dockerfile installs it. For manual Windows setup, you might need to build it yourself using [deepspeedpatcher](https://github.com/erew123/deepspeedpatcher) or try installing a precompiled wheel (see `install.bat`).

*   **LLM Model & Backend (`llm_module.py`):**
    *   Set the desired `LLM_START_MODEL` (e.g., Ollama model name or HF path) and `LLM_START_PROVIDER` in `server.py`. (The Docker setup uses `docker compose exec` to pull the specified Ollama model after starting, see Installation Step A3).
    *   Adjust system prompt in `system_prompt.txt`
*   **STT Settings (`transcribe.py`):** Modify `DEFAULT_RECORDER_CONFIG` settings at the top of the file to change Whisper model size, language, sensitivities, silence durations, etc. (The Dockerfile pre-downloads the default `base.en` model during the build).
*   **Turn Detection (`turndetect.py`):** Adjust pause duration constants in `update_settings` method.
*   **SSL (`server.py`):**
    Set `USE_SSL = True` and provide certificate/key files if HTTPS is required. You might need to adjust your `docker-compose.yml` for SSL port mapping and volume mounts for certificates.
    To generate local certificates on Windows:
    1.  Run your command prompt as administrator.
    2.  Install mkcert with Chocolatey:
        ```bat
        choco install mkcert
        ```
    3.  Install the local CA:
        ```bat
        mkcert -install
        ```
    4.  Create certificates for localhost and your local IP:
        ```bat
        mkcert 127.0.0.1 192.168.178.123
        ```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs, feature requests, or improvements.

## License

My own Codebase is MIT.
Please respect the license systems of TTS providers.