import base64
import numpy as np
from scipy.signal import resample_poly
import audioop
from typing import Optional

class ResampleOverlap:
    """
    音频流按块重采样（24kHz 到 8kHz）并进行重叠处理。
    最小化改进，专注解决机器音效问题。
    """

    def __init__(self, input_fs: int = 24000, output_fs: int = 8000, overlap_ms: int = 12):
        self.input_fs = input_fs
        self.output_fs = output_fs
        self.overlap_samples_in = int(input_fs * overlap_ms / 1000)
        self.previous_chunk = None
        self.resampled_previous_chunk = None
        self.initial_padding_samples_in = int(input_fs * overlap_ms / 1000)
        
        # 只调整关键参数
        self.kaiser_beta = 8  # 降回8，平衡质量和计算量
        self.window = ('kaiser', self.kaiser_beta)

    def get_base64_chunk(self, chunk: bytes) -> str:
        if not chunk:
            return ""

        audio_int16 = np.frombuffer(chunk, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0

        if self.previous_chunk is None:
            padded_audio_float = np.concatenate([
                np.zeros(self.initial_padding_samples_in, dtype=np.float32),
                audio_float
            ])
            downsampled = resample_poly(
                padded_audio_float,
                self.output_fs,
                self.input_fs,
                window=self.window
            )
            padding_samples_out = int(self.initial_padding_samples_in * self.output_fs / self.input_fs)
            clean_downsampled = downsampled[padding_samples_out:]

        else:
            combined = np.concatenate([self.previous_chunk[-self.overlap_samples_in:], audio_float])
            downsampled = resample_poly(
                combined,
                self.output_fs,
                self.input_fs,
                window=self.window
            )
            skip_samples = int(self.overlap_samples_in * self.output_fs / self.input_fs)
            clean_downsampled = downsampled[skip_samples:]

        self.previous_chunk = audio_float
        self.resampled_previous_chunk = downsampled

        # 只做必要的处理
        clipped = np.clip(clean_downsampled, -1.0, 1.0)
        
        # 添加非常轻微的平滑，减少量化噪声
        if len(clipped) > 2:
            smoothed = np.copy(clipped)
            smoothed[1:-1] = 0.8 * clipped[1:-1] + 0.1 * clipped[:-2] + 0.1 * clipped[2:]
            clipped = smoothed
        
        int16_audio = (clipped * 32767.0).astype(np.int16).tobytes()
        ulaw_audio = audioop.lin2ulaw(int16_audio, 2)
        return base64.b64encode(ulaw_audio).decode("utf-8")

    def flush_base64_chunk(self) -> Optional[str]:
        if self.resampled_previous_chunk is not None and self.resampled_previous_chunk.size > 0:
            total_len = len(self.resampled_previous_chunk)
            start = int(total_len * 1 / 3)
            final_chunk = self.resampled_previous_chunk[start:]

            clipped = np.clip(final_chunk, -1.0, 1.0)
            
            # 相同的轻微平滑
            if len(clipped) > 2:
                smoothed = np.copy(clipped)
                smoothed[1:-1] = 0.8 * clipped[1:-1] + 0.1 * clipped[:-2] + 0.1 * clipped[2:]
                clipped = smoothed
            
            int16_audio = (clipped * 32767.0).astype(np.int16).tobytes()
            ulaw_audio = audioop.lin2ulaw(int16_audio, 2)
            self.resampled_previous_chunk = None
            return base64.b64encode(ulaw_audio).decode("utf-8")
        return None