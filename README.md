
# Real-Time AI Voice Chat 🎤💬🧠🔊

**Have a natural, spoken conversation with an AI!**  

This project lets you chat with a Large Language Model (LLM) using just your voice, receiving spoken responses in near real-time. Think of it as your own digital conversation partner.

https://github.com/user-attachments/assets/16cc29a7-bec2-4dd0-a056-d213db798d8f

*(early preview - first reasonably stable version)*

## What's Under the Hood?

A sophisticated client-server system built for low-latency interaction:

1.  🎙️ **Capture:** Your voice is captured by your browser.
2.  ➡️ **Stream:** Audio chunks are whisked away via WebSockets to a Python backend.
3.  ✍️ **Transcribe:** `RealtimeSTT` rapidly converts your speech to text.
4.  🤔 **Think:** The text is sent to an LLM (like Ollama or OpenAI) for processing.
5.  🗣️ **Synthesize:** The AI's text response is turned back into speech using `RealtimeTTS`.
6.  ⬅️ **Return:** The generated audio is streamed back to your browser for playback.
7.  🔄 **Interrupt:** Jump in anytime! The system handles interruptions gracefully.

## Key Features ✨

*   **Fluid Conversation:** Speak and listen, just like a real chat.
*   **Real-Time Feedback:** See partial transcriptions and AI responses as they happen.
*   **Low Latency Focus:** Optimized architecture using audio chunk streaming.
*   **Smart Turn-Taking:** Dynamic silence detection (`turndetect.py`) adapts to the conversation pace.
*   **Flexible AI Brains:** Pluggable LLM backends (Ollama default, OpenAI support via `llm_module.py`).
*   **Customizable Voices:** Choose from different Text-to-Speech engines (Kokoro, Coqui, Orpheus via `audio_module.py`).
*   **Web Interface:** Clean and simple UI using Vanilla JS and the Web Audio API.
*   **Dockerized Deployment:** Recommended setup using Docker Compose for easier dependency management.

## Technology Stack 🛠️

*   **Backend:** Python < 3.13, FastAPI
*   **Frontend:** HTML, CSS, JavaScript (Vanilla JS, Web Audio API, AudioWorklets)
*   **Communication:** WebSockets
*   **Containerization:** Docker, Docker Compose
*   **Core AI/ML Libraries:**
    *   `RealtimeSTT` (Speech-to-Text)
    *   `RealtimeTTS` (Text-to-Speech)
    *   `transformers` (Turn detection, Tokenization)
    *   `torch` / `torchaudio` (ML Framework)
    *   `ollama` / `openai` (LLM Clients)
*   **Audio Processing:** `numpy`, `scipy`

## Before You Dive In: Prerequisites 🏊‍♀️

This project leverages powerful AI models, which have some requirements:

*   **Operating System:**
    *   **Docker:** Linux is recommended for the best GPU integration with Docker.
    *   **Manual:** The provided script (`install.bat`) is for Windows. Manual steps are possible on Linux/macOS but may require more troubleshooting (especially for DeepSpeed).
*   **🐍 Python:** 3.9 or higher (if setting up manually).
*   **🚀 GPU:** **A powerful CUDA-enabled NVIDIA GPU is *highly recommended***, especially for faster STT (Whisper) and TTS (Coqui). Performance on CPU-only or weaker GPUs will be significantly slower.
    *   The setup assumes **CUDA 12.1**. Adjust PyTorch installation if you have a different CUDA version.
    *   **Docker (Linux):** Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
*   **🐳 Docker (Optional but Recommended):** Docker Engine and Docker Compose v2+ for the containerized setup.
*   **🧠 Ollama (Optional):** If using the Ollama backend *without* Docker, install it separately and pull your desired models. The Docker setup includes an Ollama service.
*   **🔑 OpenAI API Key (Optional):** If using the OpenAI backend, set the `OPENAI_API_KEY` environment variable (e.g., in a `.env` file or passed to Docker).

---

## Getting Started: Installation & Setup ⚙️

**Clone the repository first:**

```bash
git clone https://github.com/zwong91/RealtimeVoiceChat.git
cd RealtimeVoiceChat

# System dependencies (Ubuntu/Debian)
apt update
apt-get -qq -y install espeak-ng > /dev/null 2>&1
apt install curl lshw ffmpeg libopenblas-dev vim git-lfs \
    build-essential cmake libasound-dev portaudio19-dev \
    libportaudio2 -y

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env


# Create and activate Python 3.10 virtual environment named 'agent'
uv venv --python=python3.10 agent
source agent/bin/activate

curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py

# Install dependencies using uv
uv pip install -r requirements.txt

# Install ollama https://github.com/ollama/ollama/releases
(curl -fsSL https://ollama.com/install.sh | sh && ollama serve > ollama.log 2>&1) &

ollama run hf.co/bartowski/huihui-ai_Mistral-Small-24B-Instruct-2501-abliterated-GGUF:Q4_K_M

```

Now, choose your adventure:

<details>
<summary><strong>🚀 Option A: Docker Installation (Recommended for Linux/GPU)</strong></summary>

This is the most straightforward method, bundling the application, dependencies, and even Ollama into manageable containers.

1.  **Build the Docker images:**
    *(This takes time! It downloads base images, installs Python/ML dependencies, and pre-downloads the default STT model.)*
    ```bash
    docker compose build
    ```
    *(If you want to customize models/settings in `code/*.py`, do it **before** this step!)*

2.  **Start the services (App & Ollama):**
    *(Runs containers in the background. GPU access is configured in `docker-compose.yml`.)*
    ```bash
    docker compose up -d
    ```
    Give them a minute to initialize.

3.  **(Crucial!) Pull your desired Ollama Model:**
    *(This is done *after* startup to keep the main app image smaller and allow model changes without rebuilding. Execute this command to pull the default model into the running Ollama container.)*
    ```bash
    # Pull the default model (adjust if you configured a different one in server.py)
    docker compose exec ollama ollama pull hf.co/bartowski/huihui-ai_Mistral-Small-24B-Instruct-2501-abliterated-GGUF:Q4_K_M

    # (Optional) Verify the model is available
    docker compose exec ollama ollama list
    ```

4.  **Stopping the Services:**
    ```bash
    docker compose down
    ```

5.  **Restarting:**
    ```bash
    docker compose up -d
    ```

6.  **Viewing Logs / Debugging:**
    *   Follow app logs: `docker compose logs -f app`
    *   Follow Ollama logs: `docker compose logs -f ollama`
    *   Save logs to file: `docker compose logs app > app_logs.txt`

</details>

<details>
<summary><strong>🛠️ Option B: Manual Installation (Windows Script / venv)</strong></summary>

This method requires managing the Python environment yourself. It offers more direct control but can be trickier, especially regarding ML dependencies.

**B1) Using the Windows Install Script:**

1.  Ensure you meet the prerequisites (Python, potentially CUDA drivers).
2.  Run the script. It attempts to create a venv, install PyTorch for CUDA 12.1, a compatible DeepSpeed wheel, and other requirements.
    ```batch
    install.bat
    ```
    *(This opens a new command prompt within the activated virtual environment.)*
    Proceed to the **"Running the Application"** section.

**B2) Manual Steps (Linux/macOS/Windows):**

1.  **Create & Activate Virtual Environment:**
    ```bash
    python -m venv venv
    # Linux/macOS:
    source venv/bin/activate
    # Windows:
    .\venv\Scripts\activate
    ```

2.  **Upgrade Pip:**
    ```bash
    python -m pip install --upgrade pip
    ```

3.  **Navigate to Code Directory:**
    ```bash
    cd code
    ```

4.  **Install PyTorch (Crucial Step - Match Your Hardware!):**
    *   **With NVIDIA GPU (CUDA 12.1 Example):**
        ```bash
        # Verify your CUDA version! Adjust 'cu121' and the URL if needed.
        pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 torchvision --index-url https://download.pytorch.org/whl/cu121
        ```
    *   **CPU Only (Expect Slow Performance):**
        ```bash
        # pip install torch torchaudio torchvision
        ```
    *   *Find other PyTorch versions:* [https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/)

5.  **Install Other Requirements:**
    ```bash
    pip install -r requirements.txt
    ```
    *   **Note on DeepSpeed:** The `requirements.txt` may include DeepSpeed. Installation can be complex, especially on Windows. The `install.bat` tries a precompiled wheel. If manual installation fails, you might need to build it from source or consult resources like [deepspeedpatcher](https://github.com/erew123/deepspeedpatcher) (use at your own risk). Coqui TTS performance benefits most from DeepSpeed.

</details>

---

## Running the Application ▶️

**If using Docker:**
Your application is already running via `docker compose up -d`! Check logs using `docker compose logs -f src`.

**If using Manual/Script Installation:**

1.  **Activate your virtual environment** (if not already active):
    ```bash
    # Linux/macOS: source ./agent/bin/activate
    # Windows: .\agent\Scripts\activate
    ```
2.  **Navigate to the `src` directory** (if not already there):
    ```bash
    cd src
    ```
3.  **Start the FastAPI server:**
    ```bash
    python server.py
    ```

**Accessing the Client (Both Methods):**

1.  Open your web browser to `http://localhost:8000` (or your server's IP if running remotely/in Docker on another machine).
2.  **Grant microphone permissions** when prompted.
3.  Click **"Start"** to begin chatting! Use "Stop" to end and "Reset" to clear the conversation.

---

## Configuration Deep Dive 🔧

Want to tweak the AI's voice, brain, or how it listens? Modify the Python files in the `code/` directory.

**⚠️ Important Docker Note:** If using Docker, make any configuration changes *before* running `docker compose build` to ensure they are included in the image.

*   **TTS Engine & Voice (`server.py`, `audio_module.py`):**
    *   Change `START_ENGINE` in `server.py` to `"coqui"`, `"kokoro"`, or `"orpheus"`.
    *   Adjust engine-specific settings (e.g., voice model path for Coqui, speaker ID for Orpheus, speed) within `AudioProcessor.__init__` in `audio_module.py`.
*   **LLM Backend & Model (`server.py`, `llm_module.py`):**
    *   Set `LLM_START_PROVIDER` (`"ollama"` or `"openai"`) and `LLM_START_MODEL` (e.g., `"hf.co/..."` for Ollama, model name for OpenAI) in `server.py`. Remember to pull the Ollama model if using Docker (see Installation Step A3).
    *   Customize the AI's personality by editing `system_prompt.txt`.
*   **STT Settings (`transcribe.py`):**
    *   Modify `DEFAULT_RECORDER_CONFIG` to change the Whisper model (`model`), language (`language`), silence thresholds (`silence_limit_seconds`), etc. The default `large-v3-turbo` model is pre-downloaded during the Docker build.
*   **Turn Detection Sensitivity (`turndetect.py`):**
    *   Adjust pause duration constants within the `TurnDetector.update_settings` method.
*   **SSL/HTTPS (`server.py`):**
    *   Set `USE_SSL = True` and provide paths to your certificate (`SSL_CERT_PATH`) and key (`SSL_KEY_PATH`) files.
    *   **Docker Users:** You'll need to adjust `docker-compose.yml` to map the SSL port (e.g., 443) and potentially mount your certificate files as volumes.
    <details>
    <summary><strong>Generating Local SSL Certificates (Windows Example w/ mkcert)</strong></summary>

    1.  Install Chocolatey package manager if you haven't already.
    2.  Install mkcert: `choco install mkcert`
    3.  Run Command Prompt *as Administrator*.
    4.  Install a local Certificate Authority: `mkcert -install`
    5.  Generate certs (replace `your.local.ip`): `mkcert localhost 127.0.0.1 ::1 your.local.ip`
        *   This creates `.pem` files (e.g., `localhost+3.pem` and `localhost+3-key.pem`) in the current directory. Update `SSL_CERT_PATH` and `SSL_KEY_PATH` in `server.py` accordingly. Remember to potentially mount these into your Docker container.
    </details>

---

## Q & A
/ws 在多个 websocket 连接期间，音频片段等内容可能会被覆盖。在垂直扩展这样的项目时，有很多瓶颈需要处理。
最糟糕的情况是两个或更多用户几乎同时完成他们的句子。这时会出现峰值 GPU 负载，并且在这种情况下无法保证低延迟。
此外，RealtimeSTT 和 RealtimeTTS 都无法处理多个并行请求。因此，目前每个实例实际上只能支持一个用户，否则你需要进行横向扩展。

进行垂直扩展则在每个新的 websocket 连接上初始化 AudioInputProcessor 以及其他类，
RealtimeSTT 和 RealtimeTTS 都无法处理并行请求，而这在多个用户场景中是必需的。
延迟下降将非常明显，整体性能将大大受损。即使只有一个用户，RealtimeVoiceChat 也确实需要一块相当强大的 GPU 才能顺畅运行。实时转录的垂直扩展涉及的内容非常复杂，首先，你面临的问题是来自不同客户端的音频块需要进行语音活动检测（VAD）。即使你设法通过批处理完美地并行化转录，你仍然会在 SileroVAD 或 webrtcVAD 上遇到瓶颈，，因为这两者都不支持批处理。两者都会引入延迟。因此，假设 10 个音频块并行到达，你处理 10 倍的 SileroVAD，增加的延迟将使实时处理变得不可能。
此外，批处理可能同时处理多个请求，但客户端的转录请求并不是完美同步到达的。你必须手动延迟它们，等待在特定时间窗口内到达足够的请求以形成一个批次。仅这一点就增加了延迟，破坏了客户的实时体验。
基于 faster_whisper，只能通过两种方式处理并行语音转文本：要么利用多个 GPU，要么批处理一个较大的音频输入文件,
这两种选项都不支持对多个传入请求的真正并发转录。
**推荐采用水平扩展的方法，使用多个实例，每个实例在一个独立的 GPU 上。**

24GB 显存（RTX 3090/4090) 运行当前模型的首次令牌时间TTFT 为 0.0563 秒，推理速度为 52.85 token/秒。
16GB 的话则 首次令牌时间低于 100 毫秒，速度超过 30 个令牌每秒。Holy fuck LLM: 139.02ms, TTS: 59.90ms

"Unable to load any of {libcudnn_ops.so.9.1.0, libcudnn_ops.so.9.1, libcudnn_ops.so.9, libcudnn_ops.so}" error
pip install "ctranslate2<4.5.0"

* VAD: Webrtcvad (first fast check) followed by SileroVAD (high compute verification)
* Transcription: large-v3-turbo whisper (CTranslate2) 300ms
* Turn Detection: livekit/turn-detector (qwen2.5:0.5b base model)
sentence binary classification model  20-50ms
* LLM: hf.co/bartowski/huihui-ai_Mistral-Small-24B-Instruct-2501-abliterated-GGUF:Q4_K_M (easily switchable) 140ms
* TTS: Coqui XTTSv2, switchable to Kokoro or Orpheus (this one is slower) 80ms

https://medium.com/@lonligrin/improving-voice-ai-with-a-sentence-completeness-classifier-2da6e950538a
基于 Silero-VAD 的转弯检测仅在用户停止说话后使用固定的沉默时间，然后决定“转弯结束”。这非常幼稚。
我使用实时转录，然后结合提示 whisper 在未完成的句子上添加省略号，以及上述模型来判断句子是否完整。
这远非完美，但相较于使用单纯的沉默来说是一个实质性的改进。

MPV 流 而不是 PCM
无法加载任何 {libcudnn_ops.so.9.1.0, libcudnn_ops.so.9.1, libcudnn_ops.so.9, libcudnn_ops.so}：

```bash
#安装 cuBLAS 和 cuDNN
apt install -y cuda-toolkit-12-4
https://developer.nvidia.com/cudnn-downloads
```

色情电话服务

https://github.com/KoljaB/RealtimeVoiceChat/blob/main/code/audio_module.py#L108

应用容器中创建一个子文件夹：./models/some_folder_name
 将您所需的语音文件复制到该文件夹中：config.json、model.pth、vocab.json 和
speakers_xtts.pth（您可以从 Lasinya 复制 speakers_xtts.pth，它对每个语音都是相同的。然后将 audio_module.py 中的 specific_model="Lasinya" 行更改为 specific_model="some_folder_name"。

Lasinya voice 是我使用自创的合成数据集制作的 XTTS 2.0.2 微调版本。我使用了 https://github.com/daswer123/xtts-finetune-webui 进行训练。

https://huggingface.co/sandy1990418/xtts-v2-chinese

https://github.com/idiap/coqui-ai-TTS/blob/dev/TTS/demos/xtts_ft_demo/XTTS_finetune_colab.ipynb

質量和速度的最佳平衡。

```python
import os
import random
import torch
import numpy as np
import pandas as pd

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    balanced_accuracy_score,
)

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from torch.utils.tensorboard import SummaryWriter

# -----------------------------
# Configuration
# -----------------------------
data_dir = "dataset"
complete_file = "dataset/filtered_complete_sentences.txt"
incomplete_file = "dataset/filtered_incomplete_sentences.txt"
# complete_file = os.path.join(data_dir, "complete_sentences.txt")
# incomplete_file = os.path.join(data_dir, "incomplete_sentences.txt")

model_name = "distilbert-base-uncased"
output_dir = "./better-distil-finetuned_model"

num_train_epochs = 30
batch_size = 512
learning_rate = 2e-5

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# -----------------------------
# Load and prepare the dataset
# -----------------------------
def load_sentences(file_path, label):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")
    return [(line.strip().lower(), label) for line in lines if line.strip()]

complete_data = load_sentences(complete_file, 1)    # label 1 for complete
incomplete_data = load_sentences(incomplete_file, 0) # label 0 for incomplete

all_data = complete_data + incomplete_data
random.shuffle(all_data)

df = pd.DataFrame(all_data, columns=["text", "label"])

# Verify class counts
print(f"Complete sentences: {df['label'].sum()}")
print(f"Incomplete sentences: {len(df) - df['label'].sum()}")

train_df, val_df = train_test_split(
    df, 
    test_size=0.1, 
    random_state=42, 
    stratify=df["label"]
)

tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

def tokenize_fn(examples):
    return tokenizer(
        examples["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=128
    )

train_dataset_hf = Dataset.from_pandas(train_df.reset_index(drop=True))
val_dataset_hf = Dataset.from_pandas(val_df.reset_index(drop=True))

train_dataset_hf = train_dataset_hf.map(tokenize_fn, batched=True)
val_dataset_hf = val_dataset_hf.map(tokenize_fn, batched=True)

train_dataset_hf.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset_hf.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# -----------------------------
# Model and Training Arguments
# -----------------------------
model = DistilBertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_train_epochs,
    learning_rate=learning_rate,
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",  # Use accuracy instead of the composite metric
    greater_is_better=True,
    save_total_limit=3,
    weight_decay=0.01,
    fp16=True,
    report_to="none"
)

# -----------------------------
# Composite Metric Computation
# -----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")

    # For ROC-AUC, we need the probability of the positive class
    # logits[:,1] gives the predicted logits for the positive class
    roc_auc = roc_auc_score(labels, logits[:,1])  
    mcc = matthews_corrcoef(labels, predictions)
    balanced_acc = balanced_accuracy_score(labels, predictions)

    # Composite metric (adjust weights as needed)
    composite_metric = (0.4 * f1) + (0.3 * roc_auc) + (0.2 * mcc) + (0.1 * balanced_acc)

    return {
        "accuracy": accuracy,
        "f1": f1,
        "roc_auc": roc_auc,
        "mcc": mcc,
        "balanced_accuracy": balanced_acc,
        "composite_metric": composite_metric,
    }

# -----------------------------
# Early Stopping Callback
# -----------------------------
callbacks = [
    EarlyStoppingCallback(early_stopping_patience=10)
]

# -----------------------------
# Custom Trainer for Logging
# -----------------------------
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(CustomTrainer, self).__init__(*args, **kwargs)
        self.writer = SummaryWriter(log_dir="./tensorboard_logs")

    def on_step_end(self, args, state, control, **kwargs):
        # Log gradient norms
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.writer.add_scalar("Grad Norm", total_norm, state.global_step)

        # Log learning rate
        current_lr = self._get_learning_rate()
        self.writer.add_scalar("Learning Rate", current_lr, state.global_step)
        super().on_step_end(args, state, control, **kwargs)

    def _get_learning_rate(self):
        # Assuming one parameter group
        return self.optimizer.param_groups[0]['lr']

    def on_train_end(self, args, state, control, **kwargs):
        self.writer.close()
        super().on_train_end(args, state, control, **kwargs)

# -----------------------------
# Initialize and Train
# -----------------------------
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_hf,
    eval_dataset=val_dataset_hf,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=callbacks,
)

trainer.train()
trainer.save_model(output_dir)

# # -----------------------------
# # Example Inference
# # -----------------------------
# test_sentences = [
#     "The cat sat on the mat.",
#     "Running down the street"
# ]
# test_inputs = tokenizer(
#     test_sentences, 
#     return_tensors="pt", 
#     truncation=True, 
#     padding="max_length", 
#     max_length=128
# )

# model.eval()
# with torch.no_grad():
#     outputs = model(**test_inputs)
#     preds = torch.argmax(outputs.logits, dim=1).tolist()

# print("Predictions:", preds)  # 1 for complete, 0 for incomplete

# -----------------------------
# To view TensorBoard logs:
# tensorboard --logdir=./tensorboard_logs
# -----------------------------

```

## Contributing 🤝

Got ideas or found a bug? Contributions are welcome! Feel free to open issues or submit pull requests.

## License 📜

The core codebase of this project is released under the **MIT License** (see the [LICENSE](./LICENSE) file for details).

This project relies on external specific TTS engines (like `Coqui XTTSv2`) and LLM providers which have their **own licensing terms**. Please ensure you comply with the licenses of all components you use.
