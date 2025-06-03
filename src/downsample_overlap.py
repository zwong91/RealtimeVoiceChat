import base64
import numpy as np
from scipy import signal
import audioop
from typing import Optional

class ResampleOverlap:
    """
    音频流按块重采样（24kHz 到 8kHz）处理。
    """

    def __init__(self, input_fs: int = 24000, output_fs: int = 8000):
        self.input_fs = input_fs
        self.output_fs = output_fs

    def get_base64_chunk(self, chunk: bytes) -> str:
        if not chunk:
            return ""
        # Step 1: bytes → int16
        pcm_array = np.frombuffer(chunk, dtype=np.int16)

        # Step 2: int16 → float32 for resampling
        audio_float = pcm_array.astype(np.float32) / 32768.0

        # Step 3: Resample 24kHz → 8kHz using polyphase
        resampled_float = resample_poly(audio_float, self.output_fs, self.input_fs)

        # Step 4: float32 → int16
        resampled_int16 = np.clip(resampled_float * 32768.0, -32768, 32767).astype(np.int16)

        # Step 5: int16 → bytes
        resampled_bytes = resampled_int16.tobytes()

        # Step 6: PCM 16-bit → μ-law (G.711)
        ulaw_bytes = audioop.lin2ulaw(resampled_bytes, 2)

        # Step 7: μ-law bytes → base64
        return base64.b64encode(ulaw_bytes).decode("utf-8")
