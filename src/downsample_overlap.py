import numpy as np
import base64
import audioop
from scipy.signal import resample_poly
from typing import Optional


class ResampleOverlap:
    """
    实现渐进式多段重叠拼接的音频重采样处理。
    用于将高采样率（如 24kHz）实时音频块转换为 8kHz 并保证拼接平滑。
    """

    def __init__(self, input_fs: int = 24000, output_fs: int = 8000,
                 overlap_ms: int = 4, fade_ms: int = 4):
        self.input_fs = input_fs
        self.output_fs = output_fs
        self.overlap_samples_in = int(overlap_ms * input_fs / 1000)
        self.fade_samples = int(fade_ms * output_fs / 1000)
        self.window = ("kaiser", 14.0)

        self.previous_chunk: Optional[np.ndarray] = None
        self.previous_overlap: Optional[np.ndarray] = None
        self.previous_tail: Optional[np.ndarray] = None

    def get_base64_chunk(self, chunk: bytes) -> str:
        if not chunk:
            return ""

        # 解码 PCM int16 到 float32
        audio_int16 = np.frombuffer(chunk, dtype=np.int16)
        if audio_int16.size == 0:
            return ""
        audio_float = audio_int16.astype(np.float32) / 32768.0

        # 拼接上一次尾部形成连续输入块
        if self.previous_chunk is None:
            combined = np.concatenate([
                np.zeros(self.overlap_samples_in, dtype=np.float32),
                audio_float
            ])
        else:
            combined = np.concatenate([
                self.previous_chunk[-self.overlap_samples_in:],
                audio_float
            ])

        # 重采样
        downsampled = resample_poly(
            combined, self.output_fs, self.input_fs, window=self.window
        )

        total_samples_out = len(downsampled)
        overlap_samples_out = int(self.overlap_samples_in * self.output_fs / self.input_fs)

        if total_samples_out <= overlap_samples_out:
            return ""

        overlap = downsampled[:overlap_samples_out]
        core = downsampled[overlap_samples_out:]

        # 淡入融合
        if self.previous_overlap is not None and len(overlap) == len(self.previous_overlap):
            fade = np.linspace(0, 1, len(overlap))
            fused_overlap = (1 - fade) * self.previous_overlap + fade * overlap
        else:
            fused_overlap = overlap

        # 拼接：融合后的 overlap + 当前 core
        output = np.concatenate([fused_overlap, core])

        # 更新状态
        self.previous_chunk = audio_float
        self.previous_overlap = overlap.copy()

        # 保存最后一段用于 flush
        if len(output) >= self.fade_samples:
            self.previous_tail = output[-self.fade_samples:].copy()
        else:
            self.previous_tail = output.copy()

        # μ-law 编码输出 base64
        return self._encode_ulaw_base64(output)

    def flush_base64_chunk(self) -> Optional[str]:
        if self.previous_tail is None or self.previous_tail.size == 0:
            return None

        final_chunk = self.previous_tail.copy()
        fade_out_len = min(len(final_chunk), self.fade_samples)
        if fade_out_len > 0:
            fade_out = np.linspace(1, 0, fade_out_len)
            final_chunk[-fade_out_len:] *= fade_out

        self.previous_tail = None
        self.previous_overlap = None
        return self._encode_ulaw_base64(final_chunk)

    def _encode_ulaw_base64(self, audio_float: np.ndarray) -> str:
        clipped = np.clip(audio_float, -0.999, 0.999)
        int16_audio = np.round(clipped * 32767.0).astype(np.int16).tobytes()
        ulaw_audio = audioop.lin2ulaw(int16_audio, 2)
        return base64.b64encode(ulaw_audio).decode("utf-8")
