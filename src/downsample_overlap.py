import base64
import numpy as np
from scipy.signal import resample_poly
import audioop
from typing import Optional

class ResampleOverlap:
    """
    实现音频流的高质量降采样（24kHz 到 8kHz），加入重叠处理与噪声门限。

    支持 PCM 输出（默认），也可以启用 u-law 编码（可选）。
    """

    def __init__(self, input_fs: int = 24000, output_fs: int = 8000,
                 overlap_ms: int = 10, use_ulaw: bool = True,
                 noise_gate_db: float = -40.0):
        """
        参数:
            input_fs: 输入采样率（Hz）
            output_fs: 输出采样率（Hz）
            overlap_ms: 每块重叠时间（毫秒）
            use_ulaw: 是否使用 u-law 编码输出
            noise_gate_db: 噪声门限（单位 dBFS，低于此值将被静音）
        """
        self.input_fs = input_fs
        self.output_fs = output_fs
        self.use_ulaw = use_ulaw
        self.noise_gate_db = noise_gate_db

        self.overlap_samples_in = int(input_fs * overlap_ms / 1000)
        self.previous_chunk = None
        self.resampled_previous_chunk = None
        self.initial_padding_samples_in = int(input_fs * 4 / 1000)  # 初始 4ms 零填充

    def _apply_noise_gate(self, signal: np.ndarray) -> np.ndarray:
        """
        将低于门限的信号静音。
        """
        if signal.size == 0:
            return signal
        rms = np.sqrt(np.mean(signal**2))
        if rms == 0:
            return np.zeros_like(signal)

        dbfs = 20 * np.log10(rms / 32768)
        if dbfs < self.noise_gate_db:
            return np.zeros_like(signal)
        return signal

    def get_base64_chunk(self, chunk: bytes) -> str:
        """
        处理单个 PCM chunk，返回 base64 编码的结果。

        参数:
            chunk: bytes 格式的 16-bit PCM 音频块（采样率 24kHz）

        返回:
            Base64 编码后的 PCM 或 u-law 音频（采样率 8kHz）
        """
        if not chunk:
            return ""

        audio_int16 = np.frombuffer(chunk, dtype=np.int16)

        # First chunk: add zero padding
        if self.previous_chunk is None:
            pad = np.zeros(self.initial_padding_samples_in, dtype=np.int16)
            audio_int16 = np.concatenate([pad, audio_int16])

        # Overlap with previous chunk
        if self.previous_chunk is not None:
            audio_int16 = np.concatenate([
                self.previous_chunk[-self.overlap_samples_in:],
                audio_int16
            ])

        # Downsample (with high-quality Kaiser window)
        audio_downsampled = resample_poly(
            audio_int16,
            up=self.output_fs,
            down=self.input_fs,
            window=('kaiser', 14.0)
        ).astype(np.int16)

        # 噪声门处理
        audio_downsampled = self._apply_noise_gate(audio_downsampled)

        # 分离可返回和保留的部分（前1/3为重叠）
        third = len(audio_downsampled) // 3
        if third == 0:
            return ""

        output_segment = audio_downsampled[third:]
        self.resampled_previous_chunk = audio_downsampled
        self.previous_chunk = audio_int16

        if self.use_ulaw:
            ulaw_bytes = audioop.lin2ulaw(output_segment.tobytes(), 2)
            return base64.b64encode(ulaw_bytes).decode('utf-8')
        else:
            return base64.b64encode(output_segment.tobytes()).decode('utf-8')

    def flush_base64_chunk(self) -> Optional[str]:
        """
        返回最后剩余的重采样块（最后 2/3），并清空状态。
        """
        if self.resampled_previous_chunk is not None:
            third = len(self.resampled_previous_chunk) // 3
            final_segment = self.resampled_previous_chunk[:third * 2]
            final_segment = self._apply_noise_gate(final_segment)

            # 清除状态
            self.previous_chunk = None
            self.resampled_previous_chunk = None

            if self.use_ulaw:
                ulaw_bytes = audioop.lin2ulaw(final_segment.tobytes(), 2)
                return base64.b64encode(ulaw_bytes).decode('utf-8')
            else:
                return base64.b64encode(final_segment.tobytes()).decode('utf-8')

        return None
