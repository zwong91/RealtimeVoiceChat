import asyncio
import logging
from typing import Optional, Callable
import numpy as np
from colors import Colors
from scipy.signal import resample_poly
from transcribe import TranscriptionProcessor

logger = logging.getLogger(__name__)


class AudioInputProcessor:
    """Processes audio input for transcription and interruption detection."""
    
    _MIN_NOISE_THRESHOLD = 300
    _MAX_NOISE_THRESHOLD = 30000
    _RESAMPLE_RATIO = 3  # 48kHz to 16kHz conversion
    _THRESHOLD_DELAY = 1  # Seconds delay for threshold determination
    _MIN_VAD_SAMPLES = 512  # at 16 kHz, 512 frames is safe for Silero

    def __init__(
            self,
            language: str = "en",
            is_orpheus: bool = False,
        ) -> None:
        self.last_partial_text: Optional[str] = None
        self.transcriber = TranscriptionProcessor(
            language,
            on_recording_start_callback=self._on_recording_start,
            is_orpheus=is_orpheus,
        )
        self.transcription_task = asyncio.create_task(self._run_transcription_loop())
        
        self.realtime_callback: Optional[Callable[[str], None]] = None
        self.recording_start_callback: Optional[Callable[[str], None]] = None
        self.interrupted = False
        
        self._setup_callbacks()

    def _on_recording_start(self) -> None:
        if self.recording_start_callback:
            self.recording_start_callback()

    def _setup_callbacks(self) -> None:
        """Configure real-time transcription and threshold detection callbacks."""
        def partial_transcript_callback(text: str) -> None:
            if text != self.last_partial_text:
                self.last_partial_text = text
                if self.realtime_callback:
                    self.realtime_callback(text)

        self.transcriber.realtime_transcription_callback = partial_transcript_callback

    async def _run_transcription_loop(self) -> None:
        """Continuous loop for handling transcription in async context."""
        while True:
            try:
                await asyncio.to_thread(self.transcriber.transcribe_loop)
            except Exception as e:
                logger.error(f"Transcription loop error: {e}")

    def process_audio_chunk(self, raw_bytes: bytes) -> np.ndarray:
        """Convert raw audio to 16kHz 16-bit PCM format."""
        raw_audio = np.frombuffer(raw_bytes, dtype=np.int16)
        
        if np.max(np.abs(raw_audio)) == 0:
            return np.zeros(len(raw_audio) // self._RESAMPLE_RATIO, dtype=np.int16)

        # Normalize and resample audio
        normalized = raw_audio.astype(np.float32) / 32767.0
        resampled = resample_poly(normalized, 1, self._RESAMPLE_RATIO)
        return np.clip(resampled * 32767, -32768, 32767).astype(np.int16)

    async def process_chunk_queue(self, audio_queue: asyncio.Queue) -> None:
        """Process incoming audio stream and detect interruptions."""
        while True:
            try:
                audio_data = await audio_queue.get()
                if audio_data is None:
                    break  # Termination signal

                pcm_data = audio_data.pop("pcm")

                processed = self.process_audio_chunk(pcm_data)
                if processed.size == 0:
                    continue

                if not self.interrupted:
                    self.transcriber.feed_audio(processed.tobytes(), audio_data)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Audio processing error: {e}", exc_info=True)

    def shutdown(self) -> None:
        """Cleanup resources and stop background tasks."""
        self.transcriber.shutdown()
        self.transcription_task.cancel()
