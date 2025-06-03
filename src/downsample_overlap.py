import base64
import numpy as np
from scipy.signal import resample_poly
import audioop
from typing import Optional

class ResampleOverlap:
    """
    音频流按块重采样（24kHz 到 8kHz）处理。
    """

    def __init__(self, input_fs: int = 24000, output_fs: int = 8000, chunk_ms=10, overlap_ratio=0.5):
        self.input_fs = input_fs
        self.output_fs = output_fs
        self.chunk_samples = int(input_fs * chunk_ms / 1000)
        self.overlap_samples = int(self.chunk_samples * overlap_ratio)
        self.prev_chunk = None

    def get_base64_chunk(self, chunk: bytes) -> str:
        if not chunk:
            return ""
        # Step 1: bytes -> int16
        current_chunk = np.frombuffer(chunk, dtype=np.int16)

        # Step 2: 拼接上一帧末尾（overlap）
        if self.prev_chunk is not None:
            input_block = np.concatenate([
                self.prev_chunk[-self.overlap_samples:],  # 上一帧末尾
                current_chunk[:self.chunk_samples - self.overlap_samples]  # 当前帧前段
            ])
        else:
            input_block = current_chunk[:self.chunk_samples]
        
        # 更新 prev_chunk 缓存
        self.prev_chunk = current_chunk

        # Step 3: int16 -> float32
        audio_float = input_block.astype(np.float32) / 32768.0

        # Step 4: Resample
        resampled = resample_poly(audio_float, self.output_fs, self.input_fs)

        # Step 5: float32 -> int16
        resampled_int16 = np.clip(resampled * 32768.0, -32768, 32767).astype(np.int16)

        # Step 6: int16 -> bytes
        resampled_bytes = resampled_int16.tobytes()

        # Step 7: PCM -> μ-law
        ulaw_bytes = audioop.lin2ulaw(resampled_bytes, 2)

        # Step 8: μ-law -> base64
        return base64.b64encode(ulaw_bytes).decode("utf-8")
