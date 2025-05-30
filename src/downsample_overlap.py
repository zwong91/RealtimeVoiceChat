import base64
import numpy as np
from scipy.signal import resample_poly
import audioop
from typing import Optional

class ResampleOverlapUlaw:
    """
    实现音频流按块重采样（24kHz 到 8kHz）并进行重叠处理。

    该类适用于处理连续的音频块，通过 scipy 的 resample_poly 进行降采样，
    并使用 u-law 编码和 Base64 编码返回结果。它内部维护状态以实现块之间的重叠处理。
    """

    def __init__(self, input_fs: int = 24000, output_fs: int = 8000, overlap_ms: int = 10):
        """
        初始化音频重采样器。

        参数:
            input_fs: 输入采样率（默认为 24kHz）
            output_fs: 输出采样率（默认为 8kHz）
            overlap_ms: 块之间的重叠窗口大小（毫秒，默认 10ms）
        """
        self.input_fs = input_fs
        self.output_fs = output_fs
        self.overlap_ms = overlap_ms

        self.overlap_samples_in = int(input_fs * overlap_ms / 1000)
        self.overlap_samples_out = int(output_fs * overlap_ms / 1000)

        self.initial_padding_samples_in = self.overlap_samples_in  # 用于首帧补零
        self.previous_chunk = None  # 上一个原始块
        self.resampled_previous_chunk = None  # 上一个重采样结果

    def get_base64_chunk(self, chunk: bytes) -> str:
        """
        处理一块音频数据并返回编码后的 Base64 字符串。

        参数:
            chunk: 原始 PCM 16 位单声道音频数据（bytes）

        返回:
            base64 编码的 u-law 音频字符串
        """
        if not chunk:
            return ""

        # 转换为 float32 数组
        audio_int16 = np.frombuffer(chunk, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0

        # 第一个块：补零处理
        if self.previous_chunk is None:
            padded_audio_float = np.concatenate([
                np.zeros(self.initial_padding_samples_in, dtype=np.float32),
                audio_float
            ])
        else:
            # 拼接上一个块的尾部（用于重叠）
            prev_tail = self.previous_chunk[-self.overlap_samples_in:]
            padded_audio_float = np.concatenate([
                prev_tail.astype(np.float32) / 32768.0,
                audio_float
            ])

        # 重采样
        resampled = resample_poly(padded_audio_float, self.output_fs, self.input_fs)

        if self.resampled_previous_chunk is None:
            # 第一次：跳过重采样后对应的 padding 部分
            padding_samples_out = int(self.initial_padding_samples_in * self.output_fs / self.input_fs)
            clean_resampled = resampled[padding_samples_out:]
        else:
            # 拼接上次保留的末尾，与这次的新数据重叠部分分开
            clean_resampled = np.concatenate([
                self.resampled_previous_chunk[-self.overlap_samples_out:],  # 保留上次尾部
                resampled[self.overlap_samples_out:]  # 跳过与上次重叠的前缀
            ])

        # 更新缓存
        self.previous_chunk = audio_int16
        self.resampled_previous_chunk = resampled

        # 限制范围 [-1, 1]，并转为 int16
        clean_resampled = np.clip(clean_resampled, -1.0, 1.0)
        resampled_int16 = (clean_resampled * 32767).astype(np.int16)

        # u-law 编码
        ulaw_data = audioop.lin2ulaw(resampled_int16.tobytes(), 2)

        # Base64 编码
        return base64.b64encode(ulaw_data).decode("ascii")

    def flush_base64_chunk(self) -> Optional[str]:
        """
        输出最后剩余的重采样音频段，并清理状态。

        通常用于处理完所有输入音频块后，返回最后一块中未输出的部分。

        返回:
            Base64 编码的 u-law 音频字符串，或 None（无剩余数据）
        """
        if self.resampled_previous_chunk is not None and self.resampled_previous_chunk.size > self.overlap_samples_out:
            final_segment = self.resampled_previous_chunk[-self.overlap_samples_out:]

            final_segment = np.clip(final_segment, -1.0, 1.0)
            final_int16 = (final_segment * 32767).astype(np.int16)

            ulaw_data = audioop.lin2ulaw(final_int16.tobytes(), 2)
            b64 = base64.b64encode(ulaw_data).decode("ascii")
        else:
            b64 = None

        # 清除内部状态
        self.previous_chunk = None
        self.resampled_previous_chunk = None

        return b64
