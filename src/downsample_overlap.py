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
        # Step 1: 转换为 numpy array，int16
        pcm_array = np.frombuffer(chunk, dtype=np.int16)
        audio_float = pcm_array.astype(np.float32) / 32768.0

        # Step 2: 降采样到 8000 Hz
        resample_len = int(len(pcm_array) * self.output_fs / self.input_fs)
        resampled = signal.resample(pcm_array, resample_len).astype(np.int16)

        # Step 3: 转换为 bytes  # chunk size 9600, (a.k.a 24K*20ms*2)
        resampled_bytes = resampled.tobytes()

        # Step 4: PCM -> μ-law
        ulaw_data = audioop.lin2ulaw(resampled_bytes, 2)  # 2 bytes per sample (16-bit)

        return base64.b64encode(ulaw_data).decode("utf-8")
