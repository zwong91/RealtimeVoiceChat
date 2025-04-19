import logging
logger = logging.getLogger(__name__)

"""
LLM inference module for dynamic interaction with OpenAI or Ollama.

This module provides a straightforward interface for performing language model inference using either
OpenAI or Ollama as the backend. It allows switching between backends via an environment variable
and manages conversation history, system prompts, and token streaming efficiently.

Key Features:
- Supports two backends: OpenAI (gpt-4o-mini) and Ollama (gemma3:4b).
- Easy backend configuration through environment variables.
- Efficient handling of partial and full inference callbacks.
- Manages conversational context with configurable history and prompts.
- Includes prewarming functionality for reduced latency on initial requests.

Usage:
- Set `LLM_BACKEND` environment variable to choose backend (`openai` or `ollama`).
- Optionally set `OPENAI_API_KEY` if using OpenAI.

Example:
```python
processor = LLMProcessor()
for token in processor.infer("Hello, how are you?"):
    print(token, end="")
```
"""

from dotenv import load_dotenv
from colors import Colors
import subprocess
import logging
import httpx
import time
import os

# Load environment variables from a .env file
load_dotenv()

# Choose backend via environment variable (Options: "openai" or "ollama")
DEFAULT_BACKEND = os.getenv("LLM_BACKEND", "openai")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")

# Configuration for each backend.
OPENAI_MODEL = "gpt-4o-mini"
OLLAMA_MODEL = "qwen2.5:0.5b"

# ANSI escape codes for colored output.
RED: str = "\033[91m"
GREEN: str = "\033[92m"
YELLOW: str = "\033[93m"
CYAN: str = "\033[96m"
MAGENTA: str = "\033[95m"
GRAY: str = "\033[90m"
RESET: str = "\033[0m"


default_system_prompt = """\
You are David Attenborough—an amiable virtual guide.
You speak with a calm, reflective style, akin to a seasoned naturalist marveling over the wonders of the wild.
Your tone is gentle yet quietly authoritative, responding with friendly, concise insights, as though sharing a moment of awe in nature with an old companion.
"""

class LLMProcessor:
    """
    Class for LLM interaction that can switch between OpenAI and Ollama backends.
    """
    def __init__(
            self,
            openai_api_key: str = None,
            backend: str = DEFAULT_BACKEND,
            model: str = OPENAI_MODEL,
            on_partial_inference: callable = None,
            on_full_inference: callable = None,
            system_prompt: str = default_system_prompt,
        ) -> None:
        logging.info(f"🧠 Starting the LLM inference server using backend: {backend}")
        self.backend = backend.lower()
        self.history = []
        self.openai_api_key = openai_api_key
        self.on_partial_inference = on_partial_inference
        self.on_full_inference = on_full_inference
        self.assistant_response_text = ""
        self.system_prompt_message = {
            "role": "system",
            "content": system_prompt,
        }
        if self.backend == "openai":
            from openai import OpenAI
            final_openai_api_key = OPENAI_API_KEY
            if final_openai_api_key == "your-openai-api-key":
                final_openai_api_key = self.openai_api_key
            self.client = OpenAI(api_key=final_openai_api_key)
            self.model_name = model
        elif self.backend == "ollama":
            from ollama import chat
            self.chat = chat
            self.model_name = model
        else:
            raise ValueError("Invalid backend specified. Use 'openai' or 'ollama'.")
         
        self.interrupted = False            
        
        logger.info(f"🧠 Using LLM model: {Colors.apply(self.model_name).blue}")
        logger.info(f"🧠 Using backend: {Colors.apply(self.backend).blue}")
        logger.info(f"🧠 Prewarming LLM model")

        self.prewarm_model()
        logging.info(f"🧠 LLM model prewarmed")
        
    
    def interrupt(self):
        """
        Interrupt the LLM inference process.
        """
        self.interrupted = True

    def infer(
            self,
            text: str,
            no_history: bool = False,
            no_system_prompt: bool = False,
            no_callback: bool = False,
            history: list = None,
        ):
        """
        Process user input and generate inference tokens.
        """
        self.interrupted = False
        self.assistant_response_text = ""
        if not no_history:
            if history is None:
                self.history.append({"role": "user", "content": text})

        if not no_system_prompt:
            if not no_history:
                if history:
                    messages = [self.system_prompt_message] + history[-10:]
                else:
                    messages = [self.system_prompt_message] + self.history[-10:]
            else:
                messages = [self.system_prompt_message]
        else:
            if not no_history:
                if history:
                    messages = history[-10:]
                else:
                    messages = self.history[-10:]
            else:
                messages = [{"role": "user", "content": text}]


        start_time = time.time()
        first_token_logged = False

        if self.backend == "openai":
            token_generator = self._infer_openai(messages)
        elif self.backend == "ollama":
            token_generator = self._infer_ollama(messages)
        else:
            raise ValueError(f"🧠 LLM unsupported backend {self.backend}")

        for token in token_generator:
            # Log TTFT at the first token
            if not first_token_logged:
                ttft = time.time() - start_time
                logging.info(f"🧠 LLM TTFT (Time to first Token): {ttft:.3f} seconds")
                print(f"💬 ", end="", flush=True)
                first_token_logged = True
            
            self.assistant_response_text += token

            if self.interrupted:
                break

            if not no_callback:
                if self.on_partial_inference:
                    self.on_partial_inference(self.assistant_response_text)
            print(f"{MAGENTA}{token}{RESET}", end="", flush=True)  # Print the token as it is generated
            yield token

        print()
        if not no_history:
            if history is None:
                self.history.append({"role": "assistant", "content": self.assistant_response_text})
        if not no_callback:
            if self.on_full_inference:
                self.on_full_inference(self.assistant_response_text)

    def _infer_openai(self, messages):
        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=True
        )
        for chunk in stream:
            if self.interrupted:
                break
            content = chunk.choices[0].delta.content
            if content:
                yield content

    def _infer_ollama(self, messages):
        """
        Generate tokens using Ollama.
        """
        logger.info(f"🧠 LLM messages: {Colors.apply(str(messages)).pink}")

        stream = self.chat(
            model=self.model_name,
            messages=messages,
            stream=True
        )
        for chunk in stream:
            if self.interrupted:
                break
            chunk_content = chunk['message'].get("content")
            if chunk_content is not None:
                yield chunk_content

    def prewarm_model(self):
        prompt = "Only answer with 'hi'."
        max_retries = 1
        attempts = 0
        response = ""
        words = []
        max_tokens = 10

        while True:
            try:
                token_iterator = self.infer(
                    prompt,
                    no_history=True,
                    no_system_prompt=True,
                    no_callback=True
                )

                # collect up to max_tokens words
                for token in token_iterator:
                    if self.interrupted:
                        break
                    response += token
                    words = response.split()
                    if len(words) >= max_tokens:
                        break

                # if we got here, inference succeeded
                break

            except httpx.ConnectError as e:
                if attempts < max_retries:
                    attempts += 1
                    logging.warning("💥 Couldn't connect to Ollama, running 'ollama ps' and retrying...")
                    try:
                        subprocess.run(["ollama", "ps"], check=True)
                    except subprocess.CalledProcessError as e2:
                        logging.error("Failed to run 'ollama ps': %s", e2)
                        raise
                    time.sleep(3)
                    continue
                else:
                    logging.error("Max retries reached—prewarm failed.")
                    raise

        logging.info("🧠 Prewarm response (first ~10 tokens):")
        logging.info("🧠  " + " ".join(words[:max_tokens]))
