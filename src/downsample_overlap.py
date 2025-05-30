import base64
import numpy as np
from scipy.signal import resample_poly
from typing import Optional
import audioop # For µ-law conversion
import math # For math.ceil

class DownsampleMuLawOverlap:
    """
    高效处理分块音频下采样到 8kHz µ-law 格式，并有效管理重叠。

    这个类设计用于处理连续的 24kHz PCM16 音频块。它使用 scipy.signal.resample_poly
    将音频下采样到 8kHz，并通过精确的重叠处理来最大程度地减少边界伪影和噪音。
    处理后的 8kHz 音频会转换为 µ-law 格式，并以 Base64 编码字符串返回。
    类内部维护状态以确保跨调用之间重叠处理的正确性。
    """
    SOURCE_RATE = 24000
    TARGET_RATE = 8000
    
    # 定义重叠长度（以 24kHz 采样点数计算）。
    # 增加重叠长度是减少噪音和不平滑现象的关键一步，因为这能给下采样滤波器更长的过渡区域。
    # 50ms 可能偏短，建议从 100ms 甚至 200ms 尝试，具体取决于 resample_poly 内部滤波器的特性。
    # 让我们先设定为 100ms 作为改进后的起点。
    OVERLAP_DURATION_MS = 100 # 100毫秒
    OVERLAP_SAMPLES_SOURCE = int(OVERLAP_DURATION_MS * SOURCE_RATE / 1000)

    def __init__(self):
        """
        初始化 DownsampleMuLawOverlap 处理器。

        设置内部状态，用于存储上一个 24kHz 原始音频块的尾部，以处理后续块的重叠。
        """
        # 存储上一个 24kHz 原始音频块的尾部数据，用于和下一个块做重叠
        self.overlap_buffer_24kHz: Optional[np.ndarray] = None 
        # 用于跟踪 flush 时是否还有数据，以及上一个块是否有效
        self._last_processed_chunk_size_8kHz: int = 0

    def get_base64_chunk(self, chunk: bytes) -> str:
        """
        处理输入的音频块，下采样到 8kHz µ-law，并返回相应的 Base64 编码片段。

        将原始的 PCM 16-bit 有符号整数字节转换为 float32 numpy 数组并归一化。
        它会将当前块与上一个块的重叠部分拼接，然后对合并后的数据进行下采样。
        从下采样结果中精确截取对应当前块的有效部分，并更新重叠缓冲区以供下次调用。
        最后，将提取出的 8kHz 音频转换为 µ-law 字节并 Base64 编码返回。

        Args:
            chunk: 原始音频数据字节 (预期为 PCM 16-bit 有符号整数，24kHz)。

        Returns:
            一个 Base64 编码的字符串，代表下采样后的 8kHz µ-law 音频片段。
            如果输入块为空，则返回空字符串。
        """
        # 优雅地处理空输入块
        if not chunk:
            return ""

        audio_int16 = np.frombuffer(chunk, dtype=np.int16)
        if audio_int16.size == 0:
            return ""

        # 将 16-bit PCM 转换为归一化的 float32，范围 [-1.0, 1.0]
        current_audio_float = audio_int16.astype(np.float32) / 32768.0

        # 如果存在历史重叠数据，则将其与当前音频块拼接起来
        if self.overlap_buffer_24kHz is not None and self.overlap_buffer_24kHz.size > 0:
            # 拼接历史重叠和当前音频数据
            combined_input_24kHz = np.concatenate((self.overlap_buffer_24kHz, current_audio_float))
        else:
            # 如果是第一个 chunk 或者重叠缓冲区为空，则直接处理当前 chunk
            combined_input_24kHz = current_audio_float

        # 对拼接后的 24kHz 数据进行下采样到 8kHz
        # resample_poly(x, up, down) 
        # 对于 24kHz -> 8kHz， up = 8000, down = 24000。
        # resample_poly 会自动约简为 up=1, down=3。
        resampled_combined_8kHz = resample_poly(combined_input_24kHz, self.TARGET_RATE, self.SOURCE_RATE)

        # 计算在 8kHz 采样率下，历史重叠数据对应的长度
        # 使用 math.ceil 确保长度计算更准确，防止微小误差导致的截断。
        num_overlap_samples_8kHz = 0
        if self.overlap_buffer_24kHz is not None and self.overlap_buffer_24kHz.size > 0:
            num_overlap_samples_8kHz = math.ceil(len(self.overlap_buffer_24kHz) * self.TARGET_RATE / self.SOURCE_RATE)
        
        # 提取输出片段：从下采样结果中，跳过历史重叠部分。
        # 我们不再使用 'expected_output_len_current_chunk_8kHz' 进行严格截取，
        # 而是取重叠部分之后的所有数据。这样可以减少因舍入误差导致的不平滑。
        output_segment_float = resampled_combined_8kHz[num_overlap_samples_8kHz:]
        
        # 更新 `overlap_buffer_24kHz`：保存当前 24kHz 音频块的尾部，用于下一次调用
        # 确保保存的重叠部分不会超过当前块的实际长度
        overlap_start_index_24kHz = max(0, len(current_audio_float) - self.OVERLAP_SAMPLES_SOURCE)
        self.overlap_buffer_24kHz = current_audio_float[overlap_start_index_24kHz:]
        
        # 记录实际输出的 8kHz 采样点数量，用于 flush 逻辑的判断
        self._last_processed_chunk_size_8kHz = len(output_segment_float)

        # 如果输出片段为空，则直接返回空字符串
        if output_segment_float.size == 0:
            return ""

        # 将 float 格式的 8kHz 音频转换为 int16 PCM
        pcm_int16_segment = (output_segment_float * 32767).astype(np.int16)
        # 将 int16 PCM 数组转换为 bytes
        pcm_int16_bytes = pcm_int16_segment.tobytes()
        # 将线性 PCM16 字节转换为 µ-law 字节 (样本宽度为 2 字节)
        ulaw_bytes = audioop.lin2ulaw(pcm_int16_bytes, 2)
        
        return base64.b64encode(ulaw_bytes).decode('utf-8')

    def flush_base64_chunk(self) -> Optional[str]:
        """
        返回所有输入块处理完成后剩余的 8kHz µ-law 音频片段。

        在所有 `get_base64_chunk` 调用完成后，此方法处理并返回
        `overlap_buffer_24kHz` 中剩余的音频数据（即最后一个输入块的尾部，
        这部分数据之前因为作为重叠而未完全输出）。
        它将这部分数据下采样到 8kHz，转换为 µ-law 格式，并以 Base64 字符串返回。
        调用此方法后，内部状态会被清空。

        Returns:
            一个 Base64 编码的字符串，代表最终的 8kHz µ-law 音频片段，
            如果没有任何数据需要刷新，则返回 None。
        """
        # 如果重叠缓冲区为空，或者上次处理的块是空的，则表示没有数据需要刷新
        if self.overlap_buffer_24kHz is None or self.overlap_buffer_24kHz.size == 0:
            return None

        # 对剩余的重叠数据进行下采样
        # 这里的 up=TARGET_RATE, down=SOURCE_RATE 保持不变
        final_segment_float = resample_poly(self.overlap_buffer_24kHz, self.TARGET_RATE, self.SOURCE_RATE)

        # 清空状态
        self.overlap_buffer_24kHz = None
        self._last_processed_chunk_size_8kHz = 0
        
        # 如果最终片段为空，则返回 None
        if final_segment_float.size == 0:
            return None

        # 转换为 µ-law 格式
        pcm_int16_segment = (final_segment_float * 32767).astype(np.int16)
        pcm_int16_bytes = pcm_int16_segment.tobytes()
        ulaw_bytes = audioop.lin2ulaw(pcm_int16_bytes, 2)
        
        return base64.b64encode(ulaw_bytes).decode('utf-8')