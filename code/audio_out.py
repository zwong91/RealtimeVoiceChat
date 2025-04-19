import os
import time
import base64
import logging
import asyncio
import wave
import numpy as np
import threading
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
        self.synthesis_running = False

        self.current_stream_chunk_size = QUICK_ANSWER_STREAM_CHUNK_SIZE

        self.working_dir = os.path.dirname(os.path.abspath(__file__))
        logger.info(f"Working directory: {self.working_dir}")

        # self.last_chunk_tail = np.zeros(64, dtype=np.float32)

        self.previous_chunk = None
        self.resampled_previous_chunk = None

        # Assuming mono audio and int16 samples:
        num_channels = 1
        sampwidth = 2  # bytes for int16
        if WRITE_FILES:
            # Open two WAV files for writing:
            # One for 24000 Hz (original) and one for 48000 Hz (upsampled).
            self.original_wave = wave.open(os.path.join(self.working_dir, "original.wav"), "wb")
            self.upsampled_wave = wave.open(os.path.join(self.working_dir, "upsampled.wav"), "wb")

            self.original_wave.setnchannels(num_channels)
            self.original_wave.setsampwidth(sampwidth)
            self.original_wave.setframerate(24000)

            self.upsampled_wave.setnchannels(num_channels)
            self.upsampled_wave.setsampwidth(sampwidth)
            self.upsampled_wave.setframerate(48000)

        # Create the Engine. Adjust arguments as needed.
        from RealtimeTTS import TextToAudioStream
        if engine == "coqui":
            from RealtimeTTS import CoquiEngine
            self.engine = CoquiEngine(
                speed=1.1,
                use_deepspeed=False,
                voice="reference_audio.wav",
                thread_count=24,
                stream_chunk_size=self.current_stream_chunk_size,
                overlap_wav_len=1024,
            )
        elif engine == "kokoro":
            voice = "af_heart"  # Default voice for Kokoro
            from RealtimeTTS import KokoroEngine
            self.engine = KokoroEngine(
                voice=voice,
                default_speed=1.2,
            )
        else:
            from RealtimeTTS import OrpheusEngine, OrpheusVoice
            # self.engine = OrpheusEngine(model="isaiahbjork/orpheus-3b-0.1-ft")
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
        #self.synthesis_running = False
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
            half_index = len(upsampled_current_chunk) // 2
            return_chunk = upsampled_current_chunk[:half_index]
        else:
            # Combine previous chunk with current data
            combined_data = np.concatenate((self.previous_chunk, audio_float))
            upsampled_float = resample_poly(combined_data, 48000, 24000)

            # Retrieve the resampled length of the previous chunk
            resampled_prev_len = len(self.resampled_previous_chunk)
            half_index_prev = resampled_prev_len // 2
            half_index_current = (len(upsampled_float) - resampled_prev_len) // 2 + resampled_prev_len

            # Return the overlapping part of the resampled data
            return_chunk = upsampled_float[half_index_prev:half_index_current]

        self.previous_chunk = audio_float
        self.resampled_previous_chunk = upsampled_current_chunk  # Store resampled data

        upsampled_audio = (return_chunk * 32767).astype(np.int16)
        upsampled_bytes = upsampled_audio.tobytes()

        if WRITE_FILES:
            self.original_wave.writeframes(chunk)
            self.upsampled_wave.writeframes(upsampled_bytes)

        # Optionally convert the upsampled data to a base64 string.
        base64_chunk = base64.b64encode(upsampled_bytes).decode('utf-8')
        return base64_chunk

    def flush_base64_chunk(self):
        # Handle the last chunk of data
        if self.previous_chunk is not None:
            upsampled_audio = (self.resampled_previous_chunk * 32767).astype(np.int16)
            upsampled_bytes = upsampled_audio.tobytes()

            if WRITE_FILES:
                self.upsampled_wave.writeframes(upsampled_bytes)

            # Optionally convert the upsampled data to a base64 string.
            base64_chunk = base64.b64encode(upsampled_bytes).decode('utf-8')

            self.previous_chunk = None
            self.resampled_previous_chunk = None

            return base64_chunk

        return None

    def start_synthesis_final_thread(self, generator) -> threading.Thread:
        """
        Start the synthesis_final process in a new thread.
        
        Args:
            generator: The generator to be passed to synthesis_final.
        
        Returns:
            The thread object running synthesis_final.
        """
        thread = threading.Thread(target=self.synthesis_final, args=(generator,))
        thread.start()
        return thread

    def start_synthesis_quick_thread(self, text: str) -> threading.Thread:
        """
        Start the synthesis_quick process in a new thread.
        
        Args:
            text: The text to be processed by synthesis_quick.
        
        Returns:
            The thread object running synthesis_quick.
        """
        thread = threading.Thread(target=self.synthesis_quick, args=(text,))
        thread.start()
        return thread

    def synthesis_final(self, generator) -> None:
        logger.info("Synthesizing final answer")

        last_log_time = time.time() - 1  # so it prints immediately on first loop

        while self.synthesis_running:
            current_time = time.time()
            if current_time - last_log_time >= 1:
                logger.info("Synthesis already running. Waiting...")
                last_log_time = current_time
            time.sleep(0.01)

        self.synthesis_running = True
        start_time = time.time()
        self.final_interrupted = False
        self.stream.feed(generator)

        if self.engine_name == "coqui" and self.current_stream_chunk_size != FINAL_ANSWER_STREAM_CHUNK_SIZE:
            self.engine.set_stream_chunk_size(FINAL_ANSWER_STREAM_CHUNK_SIZE)
            self.current_stream_chunk_size = FINAL_ANSWER_STREAM_CHUNK_SIZE

        def on_audio_chunk(chunk: bytes) -> None:
            if self.final_interrupted:
                logger.info("Final audio stream interrupted.")
                return
            if on_audio_chunk.first_call:
                on_audio_chunk.first_call = False
                logger.info(f"Final audio start. TTFT: {time.time() - start_time:.2f}s")
            
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

        time.sleep(0.1)  # Allow some time for the stream to start (more than in quick, because it's a ganerator)

        while not self.final_interrupted and self.stream.is_playing():
            time.sleep(0.001)

        time.sleep(0.1)  # Allow some time for the stream to finish
        self.synthesis_running = False

        logger.info("Final answer synthesis complete.")

    def synthesis_quick(self, text: str) -> None:
        logger.info(f"Synthesizing quick answer for: {text}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must be a non-empty string.")

        self.synthesis_running = True
        start_time = time.time()
        self.quick_interrupted = False
        self.stream.feed(self._create_generator(text))

        # switch back to quick‑chunk size if needed
        if self.engine_name == "coqui" and self.current_stream_chunk_size != QUICK_ANSWER_STREAM_CHUNK_SIZE:
            self.engine.set_stream_chunk_size(QUICK_ANSWER_STREAM_CHUNK_SIZE)
            self.current_stream_chunk_size = QUICK_ANSWER_STREAM_CHUNK_SIZE

        # jitter‑buffer state
        buffer = []
        good_streak = 0
        buffering = True
        buffered_duration = 0.0  # total playout time in buffer

        SAMPLE_RATE = 24000        # Hz
        BYTES_PER_SAMPLE = 2       # int16 mono

        def on_audio_chunk(chunk: bytes) -> None:
            nonlocal buffer, good_streak, buffering, buffered_duration

            if self.quick_interrupted:
                logger.info("Quick audio stream interrupted.")
                return

            now = time.time()
            num_samples = len(chunk) // BYTES_PER_SAMPLE
            playout = num_samples / SAMPLE_RATE

            if on_audio_chunk.first_call:
                on_audio_chunk.first_call = False
                self._quick_prev_chunk_time = now
                logger.info(f"Quick audio start. TTFT: {now - start_time:.2f}s")
                good = False
            else:
                gap = now - self._quick_prev_chunk_time
                self._quick_prev_chunk_time = now
                good = (gap <= playout)
                if good:
                    logger.info(f"👄✅ Chunk ok (gap={gap:.3f}s ≤ {playout:.3f}s)")
                else:
                    logger.warning(f"👄❌ Chunk slow (gap={gap:.3f}s > {playout:.3f}s)")

            # stash every chunk and update buffered duration
            buffer.append(chunk)
            if buffering:
                buffered_duration += playout

            if buffering:
                if good:
                    good_streak += 1
                else:
                    good_streak = 0

                # flush when 2 good in a row or ≥1s buffered
                if good_streak >= 2 or buffered_duration >= 1.0:
                    logger.info(
                        "Jitter buffer ready ("
                        f"good_streak={good_streak}, buffered_duration={buffered_duration:.3f}s); "
                        "releasing buffered chunks."
                    )
                    for buffered_chunk in buffer:
                        self.quick_answer_audio_chunks.put_nowait(buffered_chunk)
                    buffer.clear()
                    buffering = False
            else:
                # normal operation once buffer is flushed
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

        time.sleep(0.1)  # give the stream a moment to start

        # wait for stream end
        while not self.quick_interrupted and self.stream.is_playing():
            time.sleep(0.001)

        # final pause to let any last chunk arrive
        time.sleep(0.1)
        self.synthesis_running = False
        logger.info("Quick answer synthesis complete.")

    def abort_syntheses(self) -> None:
        self.quick_interrupted = True
        self.final_interrupted = True
        self.stream.stop()

        while True:
            try:
                self.quick_answer_audio_chunks.get_nowait()
            except asyncio.QueueEmpty:
                break
        while True:
            try:
                self.final_answer_audio_chunks.get_nowait()
            except asyncio.QueueEmpty:
                break

        # self.quick_answer_audio_chunks = asyncio.Queue()
        # self.final_answer_audio_chunks = asyncio.Queue()

        self.synthesis_running = False


    def abort_synthesis_quick(self) -> None:
        self.quick_interrupted = True
        self.stream.stop()
        self.quick_answer_audio_chunks = asyncio.Queue()
        self.synthesis_running = False

    def abort_synthesis_final(self) -> None:
        self.final_interrupted = True
        self.stream.stop()
        self.final_answer_audio_chunks = asyncio.Queue()
        self.synthesis_running = False

    def close_wav_files(self):
        """Call this method to finalize and close the WAV files once synthesis is complete."""
        if WRITE_FILES:
            self.original_wave.close()
            self.upsampled_wave.close()
            logger.info("WAV files closed.")
