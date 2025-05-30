import base64
import numpy as np
from scipy.signal import resample_poly
from typing import Optional
import audioop # For µ-law conversion

class DownsampleMuLawOverlap:
    SOURCE_RATE = 24000
    TARGET_RATE = 8000
    
    # 定义一个重叠长度，以24kHz采样点数计算。
    # 50毫秒是一个合理的起始点，因为下采样过程中的滤波器通常有几十毫秒的响应。
    OVERLAP_SAMPLES_SOURCE = int(0.050 * SOURCE_RATE) # 24kHz 下 50ms 的采样点数

    def __init__(self):
        # 存储上一个24kHz原始音频块的尾部数据，用于和下一个块做重叠
        self.overlap_buffer_24kHz: Optional[np.ndarray] = None 

    def get_base64_chunk(self, chunk: bytes) -> str:
        if not chunk:
            return ""

        audio_int16 = np.frombuffer(chunk, dtype=np.int16)
        if audio_int16.size == 0:
            return ""

        # 将 16-bit PCM 转换为归一化的 float32
        current_audio_float = audio_int16.astype(np.float32) / 32768.0

        # 如果存在历史重叠数据，则将其与当前音频块拼接起来
        if self.overlap_buffer_24kHz is not None:
            # 确保拼接的重叠数据不会异常的大（例如，如果当前的 chunk 非常小）
            combined_input_24kHz = np.concatenate((self.overlap_buffer_24kHz, current_audio_float))
        else:
            # 如果是第一个 chunk，则直接处理当前 chunk
            combined_input_24kHz = current_audio_float

        # 对拼接后的 24kHz 数据进行下采样到 8kHz
        resampled_combined_8kHz = resample_poly(combined_input_24kHz, self.TARGET_RATE, self.SOURCE_RATE)

        # 计算在 8kHz 采样率下，历史重叠数据对应的长度
        num_overlap_samples_8kHz = 0
        if self.overlap_buffer_24kHz is not None:
            num_overlap_samples_8kHz = round(len(self.overlap_buffer_24kHz) * self.TARGET_RATE / self.SOURCE_RATE)
        
        # 计算当前 24kHz 音频块在 8kHz 采样率下预期应有的长度
        expected_output_len_current_chunk_8kHz = round(len(current_audio_float) * self.TARGET_RATE / self.SOURCE_RATE)

        # 提取输出片段：从下采样结果中，跳过历史重叠部分，取出当前音频块对应的新数据
        # 这里的裁剪点是关键，我们从 num_overlap_samples_8kHz 处开始取，
        # 取到期望的当前块的输出长度。
        output_segment_float = resampled_combined_8kHz[num_overlap_samples_8kHz : num_overlap_samples_8kHz + expected_output_len_current_chunk_8kHz]
        
        # 处理可能的边界情况：如果计算出的输出段为空，但resampled_combined_8kHz后面还有数据，
        # 说明是最后一个很小的chunk，直接把剩余的都输出。
        if len(output_segment_float) == 0 and len(resampled_combined_8kHz) > num_overlap_samples_8kHz:
             output_segment_float = resampled_combined_8kHz[num_overlap_samples_8kHz:]

        # 更新 `overlap_buffer_24kHz`：保存当前 24kHz 音频块的尾部，用于下一次调用
        # 这里要取 `current_audio_float` 的最后一部分作为下一次的重叠
        # 使用 min 避免 chunk 过小导致负索引
        self.overlap_buffer_24kHz = current_audio_float[-min(len(current_audio_float), self.OVERLAP_SAMPLES_SOURCE):]

        # 将 float 格式的 8kHz 音频转换为 µ-law 格式并 Base64 编码
        pcm_int16_segment = (output_segment_float * 32767).astype(np.int16)
        pcm_int16_bytes = pcm_int16_segment.tobytes()
        ulaw_bytes = audioop.lin2ulaw(pcm_int16_bytes, 2)
        
        return base64.b64encode(ulaw_bytes).decode('utf-8')

    def flush_base64_chunk(self) -> Optional[str]:
        # 如果没有剩余的重叠数据，则返回 None
        if self.overlap_buffer_24kHz is None or self.overlap_buffer_24kHz.size == 0:
            return None

        # 对剩余的重叠数据进行下采样
        final_segment_float = resample_poly(self.overlap_buffer_24kHz, self.TARGET_RATE, self.SOURCE_RATE)

        # 清空状态
        self.overlap_buffer_24kHz = None
        
        if final_segment_float.size == 0:
            return None

        # 转换为 µ-law 格式
        pcm_int16_segment = (final_segment_float * 32767).astype(np.int16)
        pcm_int16_bytes = pcm_int16_segment.tobytes()
        ulaw_bytes = audioop.lin2ulaw(pcm_int16_bytes, 2)
        
        return base64.b64encode(ulaw_bytes).decode('utf-8')