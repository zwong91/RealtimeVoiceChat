import base64
import numpy as np
from scipy.signal import resample_poly
import audioop  # 用于 u-law 编码
from typing import Optional

class ResampleOverlapUlaw:
    """
    实现音频流按块重采样（24kHz 到 8kHz）并进行重叠处理。

    该类适用于处理连续的音频块，通过 scipy 的 resample_poly 进行降采样，
    并使用 u-law 编码和 Base64 编码返回结果。它内部维护状态以实现块之间的重叠处理。
    """

    def __init__(self, input_fs: int = 24000, output_fs: int = 8000, overlap_ms: int = 4):
        self.input_fs = input_fs
        self.output_fs = output_fs
        self.overlap_samples_in = int(input_fs * overlap_ms / 1000)
        self.previous_chunk: Optional[np.ndarray] = None

        # 初始零填充样本数（用于首帧）
        self.initial_padding_samples_in = self.overlap_samples_in

    def get_base64_chunk(self, chunk: bytes) -> str:
        # 空输入处理
        if not chunk:
            return ""

        # 转为浮点 PCM 格式
        audio_int16 = np.frombuffer(chunk, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0

        # 如果是第一帧，则添加前置零填充
        if self.previous_chunk is None:
            padded_audio_float = np.concatenate([
                np.zeros(self.initial_padding_samples_in, dtype=np.float32),
                audio_float
            ])
            downsampled = resample_poly(
                padded_audio_float,
                self.output_fs,
                self.input_fs
            )

            # 计算与输入填充对应的输出样本数
            padding_samples_out = int(self.initial_padding_samples_in * self.output_fs / self.input_fs)
            clean_downsampled = downsampled[padding_samples_out:]
        else:
            # 添加与上一帧的重叠部分
            input_with_overlap = np.concatenate([self.previous_chunk[-self.overlap_samples_in:], audio_float])
            downsampled = resample_poly(input_with_overlap, self.output_fs, self.input_fs)

            # 去掉重叠部分对应的输出样本
            overlap_samples_out = int(self.overlap_samples_in * self.output_fs / self.input_fs)
            clean_downsampled = downsampled[overlap_samples_out:]

        # 更新上一帧缓存
        self.previous_chunk = audio_float

        # 转回 int16 并进行 u-law 编码
        output_int16 = (np.clip(clean_downsampled, -1.0, 1.0) * 32767.0).astype(np.int16)
        ulaw_bytes = audioop.lin2ulaw(output_int16.tobytes(), 2)

        # Base64 编码
        return base64.b64encode(ulaw_bytes).decode("utf-8")


    def flush_base64_chunk(self) -> Optional[str]:
        """
        输出最后剩余的重采样音频段，并清理状态。

        通常用于处理完所有输入音频块后，返回最后一块中未输出的部分。
        数据将被转换为 u-law 编码并以 Base64 字符串返回。

        返回:
            - Base64 编码的 u-law 音频字符串，如果之前没有数据或已清空，则返回 None。
        """
        if self.resampled_previous_chunk is None or self.resampled_previous_chunk.size == 0:
            return None

        # 取出最后 2/3 的数据（用于减少边界伪影）
        start_index = self.chunk_size_out // 3
        final_chunk = self.resampled_previous_chunk[start_index:]

        # 转换为 16-bit PCM 整型格式
        final_int16 = np.clip(final_chunk * 32768, -32768, 32767).astype(np.int16)

        # 转换为 u-law 编码
        ulaw_bytes = audioop.lin2ulaw(final_int16.tobytes(), 2)

        # 编码为 Base64
        encoded = base64.b64encode(ulaw_bytes).decode('ascii')

        # 清除内部状态
        self.previous_chunk = None
        self.resampled_previous_chunk = None

        return encoded
