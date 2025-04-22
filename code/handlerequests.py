import logging
import threading
import time
import datetime
import re
from difflib import SequenceMatcher
from tokenizers import Tokenizer
from colors import Colors
from inference import LLMProcessor

logger = logging.getLogger(__name__)

#
# Choose your model
#
MODEL = "hf.co/bartowski/huihui-ai_Mistral-Small-24B-Instruct-2501-abliterated-GGUF:Q4_K_M"
#
# Select a tokenizer close to the model you are using
# (used to trim the history to fit the model context size)
#
TOKENIZER_MODEL = "mistralai/Mistral-Small-24B-Instruct-2501"


with open("system_prompt.txt", "r", encoding="utf-8") as f:
    fast_answer_system_prompt = f.read().strip()

USE_ORPHEUS_UNCENSORED = False
# You can find orpheus uncensored GGUFs that you can load into LMstudio here: https://huggingface.co/KoljaB/mOrpheus_3B-1Base_early_preview-v1-25000_GGUF

orpheus_prompt_addon_normal = """
When expressing emotions, you are ONLY allowed to use the following exact tags (including the spaces):
" <laugh> ", " <chuckle> ", " <sigh> ", " <cough> ", " <sniffle> ", " <groan> ", " <yawn> ", and " <gasp> ".

Do NOT create or use any other emotion tags. Do NOT remove the spaces. Use these tags exactly as shown, and only when appropriate.
""".strip()

orpheus_prompt_addon_uncensored = """
When expressing emotions, you are ONLY allowed to use the following exact tags (including the spaces):
" <moans> ", " <panting> ", " <grunting> ", " <gagging sounds> ", " <chokeing> ", " <kissing noises> ", " <laugh> ", " <chuckle> ", " <sigh> ", " <cough> ", " <sniffle> ", " <groan> ", " <yawn> ", " <gasp> ".
Do NOT create or use any other emotion tags. Do NOT remove the spaces. Use these tags exactly as shown, and only when appropriate.
""".strip()

orpheus_prompt_addon = orpheus_prompt_addon_uncensored if USE_ORPHEUS_UNCENSORED else orpheus_prompt_addon_normal

class LanguageProcessor:
    def __init__(
        self,
        fast_answer_callback: callable = None,
        is_orpheus: bool = False,
    ):
        logger.info("🧠 Creating LLMProcessor...")

        system_prompt = fast_answer_system_prompt
        if is_orpheus:
            system_prompt += orpheus_prompt_addon

        self.llm_fast = LLMProcessor(
            backend="ollama",
            model=MODEL,
            system_prompt=system_prompt,
        )
        logger.info("🧠 Creating Tokenizer.")
        self.tokenizer = Tokenizer.from_pretrained(TOKENIZER_MODEL)
        max_context_size = self.tokenizer.get_vocab_size()
        self.used_context_size = max_context_size * 0.9 # reserving some tokens
        logger.info(f"🧠 Tokenizer vocab size: {Colors.MAGENTA}{max_context_size}{Colors.RESET}, using 90%: {Colors.MAGENTA}{self.used_context_size}{Colors.RESET}")

        self.history = []
        self.is_working = False
        self.is_working_sentence = ""
        self.is_working_sentence_time = 0
        self.last_fast_answer = None
        self.last_fast_request_text = None
        self.last_fast_answer_time = 0
        self.answer_available = False
        self.processing_text = ""
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.paused_generator = None
        self.fast_answer_callback = fast_answer_callback
        self.final_answer_token = None
        self.split_tokens_basic = {".", "!", "?", ",", ";", ":", "\n", "-", "。", "、"}
        self.split_tokens_strict = {".", "!", "?", "。", "、"}
        self.split_tokens = self.split_tokens_strict if is_orpheus else self.split_tokens_basic

    def _count_tokens_in_message(self, message: dict) -> int:
        """
        Counts tokens in a single message dict using the global Hugging Face tokenizer.
        """
        encoding = self.tokenizer.encode(message["content"])
        return len(encoding.ids)

    def _trim_history_to_fit_context(
        self,
        history: list,
        max_context_tokens: int = 8096,
        margin_ratio: float = 0.8
    ) -> list:
        """
        Trims the conversation history so the total token count stays
        within (margin_ratio * max_context_tokens).
        Example: if max_context_tokens=2048 and margin_ratio=0.8,
        it allows ~1638 tokens.
        """
        max_tokens_allowed = int(max_context_tokens * margin_ratio)
        trimmed = []
        total_tokens = 0

        # Work backwards to keep the most recent messages
        for msg in reversed(history):
            token_count = self._count_tokens_in_message(msg)
            if total_tokens + token_count <= max_tokens_allowed:
                trimmed.append(msg)
                total_tokens += token_count
            else:
                break

        # Reverse back to normal order
        trimmed.reverse()
        return trimmed

    def get_context(self, txt: str, min_len: int = 8, max_len: int = 120):
        """
        Return substring of txt in [min_len, max_len] ending on a split char, else None.
        """
        # splits = {".", "!", "?", ",", ";", ":", "\n", "-", "。", "、"}
        for i in range(min_len, min(len(txt), max_len) + 1):
            if txt[i - 1] in self.split_tokens:
                return txt[:i], txt[i:]
        return None, None

    def reset(self):
        """Reset all conversation history and state"""
        self.history = []
        self.last_fast_answer = None
        self.last_fast_request_text = None
        self.processing_text = ""
        self.paused_generator = None
        self.pause_overhang = None
        self.stop_event.set()
        self.pause_event.clear()
        self.llm_fast.history = []

    def get_paused_generator(self):
        def tts_generator():
            if self.pause_overhang:
                if self.final_answer_token:
                    self.final_answer_token(self.pause_overhang)
                yield self.pause_overhang
            for token in self.paused_generator:
                if self.stop_event.is_set():
                    break
                if self.final_answer_token:
                    self.final_answer_token(token)
                yield token
            if self.last_final_answer_token_sent:
                self.last_final_answer_token_sent()
            print(f"🧠 {Colors.RED}-{Colors.RESET} Paused Generator set to NONE -> finished retrieval")
            self.paused_generator = None

        return tts_generator()

    def get_full_generator(self, text: str):
        def continuous_generator():
            self.stop_event.clear()
            full_answer = ""
            for token in self._process_sentence_streaming(text):
                if self.stop_event.is_set():
                    break
                full_answer += token
                if self.final_answer_token:
                    self.final_answer_token(token)
                yield token
            if self.last_final_answer_token_sent:
                self.last_final_answer_token_sent()
            self.is_working = False

        return continuous_generator()

    def _process_sentence_streaming(self, text: str):
        # Modified version that streams tokens directly
        self.is_working = True
        trimmed_history = self._trim_history_to_fit_context(self.history.copy(), max_context_tokens=self.used_context_size)
        trimmed_history.append({"role": "user", "content": text})

        logger.info(f"🧠 {Colors.MAGENTA}Inference : _process_sentence_streaming: {text}{Colors.RESET}")
        generator = self.llm_fast.infer(
            text=text,
            history=trimmed_history,
        )

        for chunk in generator:
            if self.stop_event.is_set():
                break
            yield chunk

    def _process_sentence(self, text: str):
        self.processing_text = text

        # 1. Copy current history
        history = self.history.copy()
        # 2. Trim the history to fit model context before adding new user message
        trimmed_history = self._trim_history_to_fit_context(self.history.copy(), max_context_tokens=self.used_context_size)
        # 3. Now append the new user message
        trimmed_history.append({"role": "user", "content": text})

        # 4. Send to the LLM
        logger.info(f"🧠 {Colors.MAGENTA} Inference : _process_sentence: {text}{Colors.RESET}")
        generator = self.llm_fast.infer(
            text=text,
            history=trimmed_history,
        )

        answer = ""
        print(f"🧠 {Colors.RED}-*###*-{Colors.RESET} Paused Generator set to NONE -> NEW SENTENCE PROCESSING")
        self.paused_generator = None
        self.pause_overhang = None
        self.stop_event.clear()
        for chunk in generator:
            if self.stop_event.is_set():
                break
            while self.pause_event.is_set():
                time.sleep(0.01)

            answer += chunk
            # Clean up newlines/spaces
            answer = re.sub(r'[\r\n]+', ' ', answer)
            answer = re.sub(r'\s+', ' ', answer)
            answer = answer.replace('\\n', ' ')
            answer = re.sub(r'\s+', ' ', answer)

            context, overhang = self.get_context(answer)
            if context:
                answer = context
                print(f"🧠 {Colors.RED}-*###*-{Colors.RESET} Paused Generator set -> context found")
                self.paused_generator = generator
                self.pause_overhang = overhang

                time_taken = time.time() - self.last_fast_answer_time
                logger.info(
                    f"🧠 {Colors.GRAY}Fast answer [PART]: {answer}{Colors.RESET} in {time_taken:.2f} seconds"
                )
                self.is_working = False
                self.is_working_sentence = ""
                self.is_working_sentence_time = 0
                self.last_fast_answer = answer.strip()
                self.last_fast_answer_time = time.time()
                self.last_fast_request_text = text
                self.processing_text = ""
                return

        time_taken = time.time() - self.last_fast_answer_time
        logger.info(
            f"🧠 {Colors.GRAY}Final answer: {answer}{Colors.RESET} in {time_taken:.2f} seconds"
        )
        self.is_working = False
        self.is_working_sentence = ""
        self.is_working_sentence_time = 0
        self.last_fast_answer = answer.strip()
        self.last_fast_answer_time = time.time()
        self.last_fast_request_text = text
        self.processing_text = ""

    def process_potential_sentence(self, text: str):
        if self.is_working_sentence:
            # format the timestamp as HH:MM:SS.nn
            ts = self.is_working_sentence_time
            dt = datetime.datetime.fromtimestamp(ts)
            fraction = int((ts - int(ts)) * 100)  # two decimal places
            time_str = dt.strftime("%H:%M:%S") + f".{fraction:02d}"

            logger.info(f"🧠 {Colors.RED}Already processing a sentence: {self.is_working_sentence} from time {time_str}{Colors.RESET}")            
            return

        logger.info(f"🧠 {Colors.MAGENTA}Processing potential sentence: {text}{Colors.RESET}")
        self.is_working = True
        self.is_working_sentence = text
        self.is_working_sentence_time = time.time()
        thread = threading.Thread(target=self._process_sentence, args=(text,))
        thread.daemon = True
        thread.start()

    def _preprocess_fast_answer(self, answer: str, word__limit: int = 9999999) -> str:
        """
        Splits the answer into sentences, then returns enough sentences
        to reach at least word__limit words. Discards the rest.
        """
        sentences = re.split(r'(?<=[.?!])\s+', answer.strip())
        collected = []
        total_words = 0
        for s in sentences:
            word_count = len(s.split())
            collected.append(s)
            total_words += word_count
            if total_words >= word__limit:
                break
        return ' '.join(collected)

    def _normalize_text(self, text: str) -> str:
        """Normalize text by converting to lowercase, removing punctuation and extra whitespace."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def return_fast_answer(self, text: str):
        if self.last_fast_answer and time.time() - self.last_fast_answer_time < 1:
            normalized_text = self._normalize_text(text)
            normalized_last_text = self._normalize_text(self.last_fast_request_text)
            similarity = SequenceMatcher(None, normalized_last_text, normalized_text).ratio()
            if similarity > 0.96:
                preprocessed_answer = self._preprocess_fast_answer(self.last_fast_answer)
                logger.info(f"🧠 {Colors.CYAN}Returning cached answer: {preprocessed_answer}{Colors.RESET}")
                logger.info(
                    f"🧠 Levenstein similarity: {similarity:.2f} "
                    f"between cached text: {self.last_fast_request_text} and current text: {text}"
                )
                self.last_fast_answer = preprocessed_answer
                self.answer_available = True
                if self.fast_answer_callback:
                    self.fast_answer_callback(preprocessed_answer)
                return self.last_fast_answer
            else:
                logger.info(
                    f"🧠 {Colors.RED}Cached answer is too different, similarity: {similarity:.2f}, "
                    f"cached text: {self.last_fast_request_text}, current text: {text}{Colors.RESET}"
                )
        else:
            logger.info(f"🧠 {Colors.RED}Cached answer is too old or not available.{Colors.RESET}")

        if self.processing_text == text and self.is_working_sentence:
            last_log_time = 0  # Initialize the timestamp for the last log
            while self.is_working_sentence:
                current_time = time.time()
                if current_time - last_log_time >= 1:  # Only log if at least 1 second has passed
                    logger.info(f"🧠 {Colors.RED}Waiting for answer...{Colors.RESET}")
                    last_log_time = current_time
                time.sleep(0.001)
            preprocessed_answer = self._preprocess_fast_answer(self.last_fast_answer)
            self.answer_available = True
            if self.fast_answer_callback:
                self.fast_answer_callback(preprocessed_answer)
            return self.last_fast_answer
        else:
            if not self.is_working_sentence:
                logger.info(
                    f"🧠 {Colors.CYAN}Generating answer{Colors.RESET} "
                    "(no llm request running + cached request too old / not available)"
                )
            else:
                logger.info(
                    f"🧠 {Colors.CYAN}Generating answer{Colors.RESET} "
                    f"(llm request running + self.processing_text {self.processing_text} != text {text})"
                )

        self._process_sentence(text)
        preprocessed_answer = self._preprocess_fast_answer(self.last_fast_answer)
        logger.info(f"🧠 {Colors.CYAN}Returning generated answer: {preprocessed_answer}{Colors.RESET}")
        self.last_fast_answer = preprocessed_answer
        self.answer_available = True
        if self.fast_answer_callback:
            self.fast_answer_callback(preprocessed_answer)
        return self.last_fast_answer

    def return_fast_sentence_answer(self, text: str):
        self.answer_available = False
        thread = threading.Thread(target=self.return_fast_answer, args=(text,))
        thread.daemon = True
        thread.start()

    def shutdown(self):
        logger.info("🧠 Shutting down language processing...")
