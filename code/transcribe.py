import logging
logger = logging.getLogger(__name__)

from turndetect import strip_ending_punctuation
from difflib import SequenceMatcher
from colors import Colors
from scipy import signal
import numpy as np
import threading
import textwrap
import torch
import json
import copy
import time
import re

USE_TURN_DETECTION = True
SILERO_SENSITIVITY = 0.4
START_STT_SERVER = False

if START_STT_SERVER:
    from RealtimeSTT import AudioToTextRecorderClient
else:
    from RealtimeSTT import AudioToTextRecorder


INT16_MAX_ABS_VALUE = 32768.0
SAMPLE_RATE = 16000

if USE_TURN_DETECTION:
    from turndetect import TurnDetection

class TranscriptionProcessor:
    def __init__(
            self,
            source_language="en",
            realtime_transcription_callback=None,
            full_transcription_callback=None,
            potential_full_transcription_callback=None,
            potential_full_transcription_abort_callback=None,
            potential_sentence_end=None,
            before_final_sentence=None,
            silence_active_callback=None,
            on_recording_start_callback=None,
            local=True,
    ):
        self.source_language = source_language
        self.realtime_transcription_callback = realtime_transcription_callback
        self.full_transcription_callback = full_transcription_callback
        self.potential_full_transcription_callback = potential_full_transcription_callback
        self.potential_full_transcription_abort_callback = potential_full_transcription_abort_callback
        self.potential_sentence_end = potential_sentence_end
        self.before_final_sentence = before_final_sentence
        self.silence_active_callback = silence_active_callback
        self.on_recording_start_callback = on_recording_start_callback
        self.recorder = None
        self.is_silero_speech_active = False
        self.silero_working = False
        self.on_wakeword_detection_start = None
        self.on_wakeword_detection_end = None
        self.realtime_text = None
        self.sentence_end_cache = []
        self.potential_sentences_yielded = []
        self.stripped_partial_user_text = ""
        self.final_transcription = None
        self.shutdown_performed = False
        self.silence_time = 0
        self.silence_active = False

        if USE_TURN_DETECTION:
            logger.info(f"👂 {Colors.YELLOW}Turn detection enabled{Colors.RESET}")
            self.turn_detection = TurnDetection(
                on_new_waiting_time=self.on_new_waiting_time,
                local=local
            )

        self._create_recorder()
        self._start_silence_monitor()

    def _start_silence_monitor(self):
        def monitor():
            hot = False
            if START_STT_SERVER:
                self.silence_time = self.recorder.get_parameter("speech_end_silence_start")
            else:
                self.silence_time = self.recorder.speech_end_silence_start
            
            while not self.shutdown_performed:
                speech_end_silence_start = self.silence_time
                if self.recorder and speech_end_silence_start is not None and speech_end_silence_start != 0:
                    if START_STT_SERVER:
                        silence_waiting_time = self.recorder.get_parameter("post_speech_silence_duration")
                    else:
                        silence_waiting_time = self.recorder.post_speech_silence_duration
                    time_since_silence = time.time() - speech_end_silence_start

                    potential_sentence_end_time = 0.1
                    if potential_sentence_end_time > silence_waiting_time * 0.2:
                        potential_sentence_end_time = silence_waiting_time * 0.2

                    start_hot_condition = silence_waiting_time - 0.35
                    if start_hot_condition < 0.15:
                        start_hot_condition = 0.15

                    if potential_sentence_end_time > start_hot_condition - 0.1:
                        potential_sentence_end_time = start_hot_condition - 0.1

                    if time_since_silence > potential_sentence_end_time:
                        self.detect_potential_sentence_end(self.realtime_text)

                    hot_condition = time_since_silence > start_hot_condition
                    if hot_condition and not hot:
                        hot = True
                        print(f"{Colors.MAGENTA}HOT{Colors.RESET}")
                        if self.potential_full_transcription_callback:
                            self.potential_full_transcription_callback(self.realtime_text)
                    elif not hot_condition and hot:                        
                        if START_STT_SERVER:
                            is_recording = self.recorder.get_parameter("is_recording")
                        else:
                            is_recording = self.recorder.is_recording
                        if is_recording:
                            print(f"{Colors.CYAN}COLD{Colors.RESET}")
                            if self.potential_full_transcription_abort_callback:
                                self.potential_full_transcription_abort_callback()
                        hot = False
                elif hot:
                    # if self.recorder.is_recording:
                    if START_STT_SERVER:
                        is_recording = self.recorder.get_parameter("is_recording")
                    else:
                        is_recording = self.recorder.is_recording
                    if is_recording:
                        print(f"{Colors.CYAN}COLD{Colors.RESET}")
                        if self.potential_full_transcription_abort_callback:
                            self.potential_full_transcription_abort_callback()
                    hot = False
                time.sleep(0.001)
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()

    def on_new_waiting_time(
            self,
            waiting_time: float,
            text: str = None,
        ) -> None:
        """Handle a new waiting time.

        Args:
            waiting_time: The new waiting time in seconds.
        """
        if self.recorder:
            if START_STT_SERVER:
                post_speech_silence_duration = self.recorder.get_parameter("post_speech_silence_duration")
            else:
                post_speech_silence_duration = self.recorder.post_speech_silence_duration
            if post_speech_silence_duration != waiting_time:
                logger.info(f"👂 {Colors.GRAY}New waiting time: {Colors.RESET}{Colors.YELLOW}{waiting_time:.2f}{Colors.RESET}{Colors.GRAY} for text: {text}{Colors.RESET}")
                if START_STT_SERVER:
                    self.recorder.set_parameter("post_speech_silence_duration", waiting_time)
                else:
                    self.recorder.post_speech_silence_duration = waiting_time
        else:
            logger.info("👂 Recorder not initialized")

    def transcribe_loop(self):
        def on_final(text):
            if text is None:
                return
            if text == "":
                return
            
            self.final_transcription = text
            logger.info(f"👂 {Colors.apply('Final user text: ').green} {Colors.apply(text).yellow}")
            self.sentence_end_cache.clear()
            self.potential_sentences_yielded.clear()

            if USE_TURN_DETECTION:
                self.turn_detection.reset()            
            if self.full_transcription_callback:
                self.full_transcription_callback(text)
        self.recorder.text(on_final)

    def perform_final(self, audio_bytes):
        if self.recorder:
            if self.realtime_text is None:
                logger.info(f"👂 {Colors.RED}Final text is None{Colors.RESET}")
                self.realtime_text = ""
                
            final_transcription = self.realtime_text
            self.final_transcription = final_transcription
            logger.info(f"👂 {Colors.apply('Final user text: ').green} {Colors.apply(final_transcription).yellow}")
            self.sentence_end_cache.clear()
            self.potential_sentences_yielded.clear()

            if USE_TURN_DETECTION:
                self.turn_detection.reset()            
            if self.full_transcription_callback:
                self.full_transcription_callback(final_transcription)
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text by converting to lowercase, removing punctuation and extra whitespace."""
        # Convert to lowercase
        text = text.lower()
        # Remove all non-alphanumeric characters
        text = re.sub(r'[^a-zA-Z0-9]', '', text)
        # Remove extra whitespace and trim
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def is_basically_the_same(
            self,
            text1,
            text2,
            similarity_threshold=0.96,
        ):
        text1 = self._normalize_text(text1)
        text2 = self._normalize_text(text2)
        similarity = SequenceMatcher(None, text1, text2).ratio()
        return similarity > similarity_threshold
    
    def detect_potential_sentence_end(self, text):
        if text is None:
            return
        end_punctuations = [".", "!", "?"]
        stripped_text = text.strip()
        if any(stripped_text.endswith(p) for p in end_punctuations):
            if not stripped_text.endswith("...") and self.potential_sentence_end:
                stripped_text = self._normalize_text(stripped_text)
                now = time.time()
                # Search for an entry with the same text
                entry_found = None
                for entry in self.sentence_end_cache:
                    if self.is_basically_the_same(entry['text'], stripped_text):
                        entry_found = entry
                        break

                if entry_found is not None:
                    if now - entry_found['timestamp'] > 0.1:
                        already_yielded = False
                        for entry in self.potential_sentences_yielded:
                            if self.is_basically_the_same(entry['text'], stripped_text):
                                already_yielded = True
                                break
                        
                        if not already_yielded:
                            self.potential_sentence_end(text)
                            entry_found['timestamp'] = now  # Update the timestamp
                            self.potential_sentences_yielded.append({'text': stripped_text, 'timestamp': now})
                else:
                    # Only append if the text is not already in the cache
                    self.sentence_end_cache.append({'text': stripped_text, 'timestamp': now})

    def set_silence(self, silence_active: bool):
        if self.silence_active != silence_active:
            self.silence_active = silence_active
            print(f"&&& {Colors.MAGENTA}Silence detection callback{Colors.RESET} {Colors.YELLOW}{'activated' if silence_active else 'deactivated'}{Colors.RESET}")
            if self.silence_active_callback:
                self.silence_active_callback(silence_active)

    def get_audio_copy(self):
        full_audio_array = np.frombuffer(b''.join(self.recorder.frames), dtype=np.int16)
        full_audio = full_audio_array.astype(np.float32) / INT16_MAX_ABS_VALUE
        audio_bytes = copy.deepcopy(full_audio)
        return audio_bytes

    def _create_recorder(self):
        def start_silence_detection():
            self.set_silence(True)
            self.silence_time = time.time()
        
        def stop_silence_detection():
            self.set_silence(False)
            self.silence_time = 0
        
        def start_recording():
            self.set_silence(False)
            if self.on_recording_start_callback:
                self.on_recording_start_callback()
        
        def stop_recording():
            self.set_silence(True)

        def before_final(audio_copy):
            if self.before_final_sentence:
                self.before_final_sentence(audio_copy, self.realtime_text)
                return True

        def on_partial(text):
            if text is None:
                print(f"{Colors.RED}Partial text is None{Colors.RESET}")
                return
            self.realtime_text = text
            self.detect_potential_sentence_end(text)

            stripped_partial_user_text_new = strip_ending_punctuation(text)
            if stripped_partial_user_text_new != self.stripped_partial_user_text:
                self.stripped_partial_user_text = stripped_partial_user_text_new
                logger.info(f"👂 Partial transcription: {Colors.CYAN}{text}{Colors.RESET}")
                if self.realtime_transcription_callback:
                    self.realtime_transcription_callback(text)
                if USE_TURN_DETECTION:
                    self.turn_detection.calculate_waiting_time(
                        text = text
                    )
            else:
                logger.info(f"👂 Partial transcription: {Colors.GRAY}{text}{Colors.RESET}")                

        def _pretty(v, max_len=60):
            """Return a printable version of any value."""
            if callable(v):
                return f"[callback: {v.__name__}]"
            if isinstance(v, str):
                one_line = v.replace("\n", " ")
                return (one_line[:max_len].rstrip() + " [...]") if len(one_line) > max_len else one_line
            return v

        if START_STT_SERVER:
            logger.info(f"👂 Creating AudioToTextRecorderClient (STT CLIENT/SERVER VERSION) with params:")
        else:
            logger.info(f"👂 Creating AudioToTextRecorder (STT NATIVE VERSION) with params:")

        # 1) Collect every keyword in a plain dict

        incompletion_prompt = (
            "End incomplete sentences with ellipses.\n"
            "Examples:\n"
            "Complete: The sky is blue.\n"
            "Incomplete: When the sky...\n"
            "Complete: She walked home.\n"
            "Incomplete: Because he...\n"
        )

        incompletion_prompt = "The sky is blue. When the sky... She walked home. Because he... Today is sunny. If only I..."

        recorder_cfg = {
            "use_microphone": False,
            "spinner": False,

            # "model": "base.en",
            # "realtime_model_type": "base.en",
            # "use_main_model_for_realtime": False,
            "model": "base.en",
            #"realtime_model_type": "large-v2",
            "use_main_model_for_realtime": True,

            "language": self.source_language,
            #"silero_sensitivity": 0.1,
            "silero_sensitivity": 0.05,
            "webrtc_sensitivity": 3,
            "post_speech_silence_duration": 0.7,
            "min_length_of_recording": 0.5,
            "min_gap_between_recordings": 0,
            "enable_realtime_transcription": True,
            "realtime_processing_pause": 0.01,
            "silero_use_onnx": True,
            "silero_deactivity_detection": True,
            "early_transcription_on_silence": 0,
            "beam_size": 3,
            "beam_size_realtime": 3,
            # "batch_size": 1,
            # "realtime_batch_size": 1,
            "no_log_file": True,
            "wake_words": "jarvis",
            "wakeword_backend": "pvporcupine",
            "allowed_latency_limit": 500,
            "on_realtime_transcription_update": on_partial,
            "on_transcription_start": before_final,
            "on_turn_detection_start": start_silence_detection,
            "on_turn_detection_stop": stop_silence_detection,
            "on_recording_start": start_recording,
            "on_recording_stop": stop_recording,
            "debug_mode": True,
            "initial_prompt": incompletion_prompt,    
            "initial_prompt_realtime": incompletion_prompt,
        
            "faster_whisper_vad_filter": False,
        }

        pretty_cfg = {k: _pretty(v) for k, v in recorder_cfg.items()}

        padded = textwrap.indent(json.dumps(pretty_cfg, indent=2), "    ")
        print(Colors.apply(padded).blue)

        # 3) Instantiate the client
        if START_STT_SERVER:

            self.recorder = AudioToTextRecorderClient(**recorder_cfg)
            self.recorder.set_parameter("use_wake_words", False)

        else:
            self.recorder = AudioToTextRecorder(**recorder_cfg)
            self.recorder.use_wake_words = False

        logger.info("👂 Recorder created.")

    def feed_audio(self, chunk, audio_meta_data):
        if self.recorder:
            self.recorder.feed_audio(chunk, audio_meta_data)

    def shutdown(self):
        logger.info("👂 Shutting down recorder...")
        self.shutdown_performed = True
        if self.recorder:
            self.recorder.shutdown()
            self.recorder = None
            logger.info("👂 Recorder shutdown completed.")
