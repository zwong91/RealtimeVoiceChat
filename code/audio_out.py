import os
import time
import base64
import logging
import asyncio
import wave
import numpy as np
import threading
from huggingface_hub import hf_hub_download
from scipy.signal import resample_poly
from collections import namedtuple

START_ENGINE = "kokoro"  # Default engine to use if not specified

Silence = namedtuple("Silence", ("comma", "sentence", "default"))

ENGINE_SILENCES = {
    "coqui":   Silence(comma=0.3, sentence=0.6, default=0.3),
    "kokoro":  Silence(comma=0.3, sentence=0.6, default=0.3),
    "orpheus": Silence(comma=0, sentence=0, default=0),
}

logger = logging.getLogger(__name__)

WRITE_FILES = False  # Set to True to write WAV files for debugging.
QUICK_ANSWER_STREAM_CHUNK_SIZE = 8
FINAL_ANSWER_STREAM_CHUNK_SIZE = 30

# Coqui model download helper functions
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def ensure_lasinya_models(models_root="models", model_name="Lasinya"):
    base = os.path.join(models_root, model_name)
    create_directory(base)
    files = ["config.json", "vocab.json", "speakers_xtts.pth", "model.pth"]
    for fn in files:
        local_file = os.path.join(base, fn)
        if not os.path.exists(local_file):
            print(f"Downloading {fn} to {base}")
            hf_hub_download(
                repo_id="KoljaB/XTTS_Lasinya",
                filename=fn,
                local_dir=base
            )

class AudioOutProcessor:
    def __init__(
            self,
            engine: str = START_ENGINE,
            language: str = "en",
        ):
        self.engine_name = engine
        self.silence = ENGINE_SILENCES.get(engine, ENGINE_SILENCES["orpheus"])

        self.quick_answer_audio_chunks = asyncio.Queue()
        self.final_answer_audio_chunks = asyncio.Queue()

        self.quick_interrupted = False
        self.final_interrupted = False
        # use Event instead of bool
        self.synthesis_available = threading.Event()
        self.synthesis_available.set()


        self.current_stream_chunk_size = QUICK_ANSWER_STREAM_CHUNK_SIZE

        self.working_dir = os.path.dirname(os.path.abspath(__file__))
        logger.info(f"Working directory: {self.working_dir}")

        self.previous_chunk = None
        self.resampled_previous_chunk = None

        # Assuming mono audio and int16 samples:
        num_channels = 1
        sampwidth = 2
        if WRITE_FILES:
            self.original_wave = wave.open(os.path.join(self.working_dir, "original.wav"), "wb")
            self.upsampled_wave = wave.open(os.path.join(self.working_dir, "upsampled.wav"), "wb")

            self.original_wave.setnchannels(num_channels)
            self.original_wave.setsampwidth(sampwidth)
            self.original_wave.setframerate(24000)

            self.upsampled_wave.setnchannels(num_channels)
            self.upsampled_wave.setsampwidth(sampwidth)
            self.upsampled_wave.setframerate(48000)

        from RealtimeTTS import TextToAudioStream
        if engine == "coqui":
            ensure_lasinya_models(models_root="models", model_name="Lasinya")
            from RealtimeTTS import CoquiEngine
            self.engine = CoquiEngine(
                specific_model="Lasinya",
                local_models_path="./models",
                voice="reference_audio.wav",
                speed=1.1,
                use_deepspeed=True,
                thread_count=6,
                stream_chunk_size=self.current_stream_chunk_size,
                overlap_wav_len=1024,
            )
        elif engine == "kokoro":
            from RealtimeTTS import KokoroEngine
            self.engine = KokoroEngine(
                voice="af_heart",
                default_speed=1.2,
            )
        else:
            from RealtimeTTS import OrpheusEngine, OrpheusVoice
            self.engine = OrpheusEngine(model="isaiahbjork/orpheus-3b-0.1-ft")
            voice = OrpheusVoice("tara")
            self.engine.set_voice(voice)

        # Create a TextToAudioStream that handles streaming audio from the engine.
        self.stream = TextToAudioStream(
            self.engine,
            muted=True,
            on_audio_stream_stop=self._on_stream_stop,
            playout_chunk_size=4096,
        )

        # Pre-warm the stream.
        self.stream.feed("warm up").play(muted=True)

    def _on_stream_stop(self):
        """Callback when the audio stream stops."""
        logger.info("Audio stream stopped.")

    def _create_generator(self, text: str):
        """Create a generator that yields text once."""
        def generator():
            yield text
        return generator()

    def get_base64_chunk(self, chunk) -> str:
        """
        Process a PCM chunk, write the original and upsampled data
        to WAV files and return the upsampled data as base64.
        """
        audio_int16 = np.frombuffer(chunk, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0

        upsampled_current_chunk = resample_poly(audio_float, 48000, 24000)
        if self.previous_chunk is None:
            half = len(upsampled_current_chunk) // 2
            part = upsampled_current_chunk[:half]
        else:
            combined = np.concatenate((self.previous_chunk, audio_float))
            up = resample_poly(combined, 48000, 24000)
            prev_len = len(self.resampled_previous_chunk)
            h_prev = prev_len // 2
            h_cur = (len(up) - prev_len) // 2 + prev_len
            part = up[h_prev:h_cur]

        self.previous_chunk = audio_float
        self.resampled_previous_chunk = upsampled_current_chunk

        pcm = (part * 32767).astype(np.int16).tobytes()
        if WRITE_FILES:
            self.original_wave.writeframes(chunk)
            self.upsampled_wave.writeframes(pcm)
        return base64.b64encode(pcm).decode('utf-8')

    def flush_base64_chunk(self):
        """Handle the last chunk of data"""
        if self.previous_chunk is not None:
            pcm = (self.resampled_previous_chunk * 32767).astype(np.int16).tobytes()
            if WRITE_FILES:
                self.upsampled_wave.writeframes(pcm)
            self.previous_chunk = None
            self.resampled_previous_chunk = None
            return base64.b64encode(pcm).decode('utf-8')
        return None

    def start_synthesis_final_thread(self, generator) -> threading.Thread:
        """Start the synthesis_final process in a new thread."""
        t = threading.Thread(target=self.synthesis_final, args=(generator,))
        t.start()
        return t

    def start_synthesis_quick_thread(self, text: str) -> threading.Thread:
        """Start the synthesis_quick process in a new thread."""
        t = threading.Thread(target=self.synthesis_quick, args=(text,))
        t.start()
        return t

    def synthesis_final(self, generator) -> None:

        self.synthesis_available.wait()
        # now take it
        logger.info("Synthesizing final answer")
        self.synthesis_available.clear()

        start = time.time()
        self.final_interrupted = False
        self.stream.feed(generator)

        if self.engine_name == "coqui" and self.current_stream_chunk_size != FINAL_ANSWER_STREAM_CHUNK_SIZE:
            self.engine.set_stream_chunk_size(FINAL_ANSWER_STREAM_CHUNK_SIZE)
            self.current_stream_chunk_size = FINAL_ANSWER_STREAM_CHUNK_SIZE

        def on_audio_chunk(chunk: bytes):
            if self.final_interrupted:
                logger.info("Final audio stream interrupted.")
                return
            if on_audio_chunk.first_call:
                on_audio_chunk.first_call = False
                logger.info(f"Final audio start. TTFT: {time.time()-start:.2f}s")
            self.final_answer_audio_chunks.put_nowait(chunk)
        on_audio_chunk.first_call = True

        self.stream.play_async(
            log_synthesized_text=True,
            on_audio_chunk=on_audio_chunk,
            fast_sentence_fragment=False,
            muted=True,
            comma_silence_duration=self.silence.comma,
            sentence_silence_duration=self.silence.sentence,
            default_silence_duration=self.silence.default,
        )

        time.sleep(0.1)
        while not self.final_interrupted and self.stream.is_playing():
            time.sleep(0.001)
        time.sleep(0.1)
        # clear running flag
        self.synthesis_available.set()
        logger.info("Final answer synthesis complete.")

    def synthesis_quick(self, text: str) -> None:
        logger.info(f"Synthesizing quick answer for: {text}")
        if not text.strip():
            raise ValueError("Text must be a non-empty string.")
        # mark running
        self.synthesis_available.wait()
        self.synthesis_available.clear()
        start = time.time()
        self.quick_interrupted = False
        self.stream.feed(self._create_generator(text))

        if self.engine_name == "coqui" and self.current_stream_chunk_size != QUICK_ANSWER_STREAM_CHUNK_SIZE:
            self.engine.set_stream_chunk_size(QUICK_ANSWER_STREAM_CHUNK_SIZE)
            self.current_stream_chunk_size = QUICK_ANSWER_STREAM_CHUNK_SIZE

        buffer, good_streak, buffering, buf_dur = [], 0, True, 0.0
        SR, BPS = 24000, 2

        def on_audio_chunk(chunk: bytes):
            nonlocal buffer, good_streak, buffering, buf_dur
            if self.quick_interrupted:
                logger.info("Quick audio stream interrupted.")
                return
            now = time.time()
            samples = len(chunk) // BPS
            play = samples / SR

            if on_audio_chunk.first_call:
                on_audio_chunk.first_call = False
                self._quick_prev_chunk_time = now
                logger.info(f"Quick audio start. TTFT: {now-start:.2f}s")
            else:
                gap = now - self._quick_prev_chunk_time
                self._quick_prev_chunk_time = now
                if gap <= play:
                    logger.info(f"👄✅ Chunk ok (gap={gap:.3f}s ≤ {play:.3f}s)")
                    good_streak += 1
                else:
                    logger.warning(f"👄❌ Chunk slow (gap={gap:.3f}s > {play:.3f}s)")
                    good_streak = 0

            buffer.append(chunk)
            if buffering:
                buf_dur += play
                if good_streak >= 2 or buf_dur >= 1.0:
                    for c in buffer:
                        self.quick_answer_audio_chunks.put_nowait(c)
                    buffer.clear()
                    buffering = False
            else:
                self.quick_answer_audio_chunks.put_nowait(chunk)
        on_audio_chunk.first_call = True

        self.stream.play_async(
            log_synthesized_text=True,
            on_audio_chunk=on_audio_chunk,
            muted=True,
            comma_silence_duration=self.silence.comma,
            sentence_silence_duration=self.silence.sentence,
            default_silence_duration=self.silence.default,
        )

        time.sleep(0.1)
        while not self.quick_interrupted and self.stream.is_playing():
            time.sleep(0.001)
        time.sleep(0.1)
        # clear running flag
        self.synthesis_available.set()
        logger.info("Quick answer synthesis complete.")

    def abort_syntheses(self) -> None:
        self.quick_interrupted = True
        self.final_interrupted = True
        self.stream.stop()
        for q in (self.quick_answer_audio_chunks, self.final_answer_audio_chunks):
            while True:
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    break
        self.synthesis_available.set()

    def abort_synthesis_quick(self) -> None:
        self.quick_interrupted = True
        self.stream.stop()
        self.quick_answer_audio_chunks = asyncio.Queue()
        self.synthesis_available.set()

    def abort_synthesis_final(self) -> None:
        self.final_interrupted = True
        self.stream.stop()
        self.final_answer_audio_chunks = asyncio.Queue()
        self.synthesis_available.set()

    def close_wav_files(self):
        """Call this method to finalize and close the WAV files once synthesis is complete."""
        if WRITE_FILES:
            self.original_wave.close()
            self.upsampled_wave.close()
            logger.info("WAV files closed.")
