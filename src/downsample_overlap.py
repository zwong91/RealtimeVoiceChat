import base64
import numpy as np
from scipy import signal
import audioop
from typing import Optional

class ResampleOverlap:
    """
    使用简单线性插值的音频重采样，避免复杂滤波造成的杂音。
    """

    def __init__(self, input_fs: int = 24000, output_fs: int = 8000, overlap_ms: int = 20):
        self.input_fs = input_fs
        self.output_fs = output_fs
        self.downsample_factor = input_fs // output_fs  # 24000/8000 = 3
        
        self.overlap_samples_in = int(input_fs * overlap_ms / 1000)
        self.previous_chunk = None
        
        # 简单的抗混叠低通滤波器（截止频率为输出采样率的一半）
        # 使用Butterworth滤波器，比Kaiser窗更平滑
        nyquist = self.output_fs / 2
        cutoff = nyquist * 0.8  # 保守一点
        self.b, self.a = signal.butter(4, cutoff / (input_fs / 2), btype='low')

    def _simple_downsample(self, audio: np.ndarray) -> np.ndarray:
        """使用简单的线性插值下采样"""
        # 先应用抗混叠滤波
        if len(audio) > 10:  # 确保有足够的样本
            filtered = signal.filtfilt(self.b, self.a, audio)
        else:
            filtered = audio
            
        # 计算新的时间轴
        input_time = np.arange(len(filtered)) / self.input_fs
        output_samples = int(len(filtered) * self.output_fs / self.input_fs)
        output_time = np.arange(output_samples) / self.output_fs
        
        # 使用线性插值
        downsampled = np.interp(output_time, input_time, filtered)
        return downsampled

    def get_base64_chunk(self, chunk: bytes) -> str:
        if not chunk:
            return ""

        audio_int16 = np.frombuffer(chunk, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0

        if self.previous_chunk is None:
            # 第一块
            processed_audio = audio_float
        else:
            # 使用重叠拼接
            overlap_part = self.previous_chunk[-self.overlap_samples_in:]
            
            # 简单的交叉淡化
            if len(audio_float) >= self.overlap_samples_in:
                fade_samples = min(self.overlap_samples_in, len(overlap_part))
                if fade_samples > 0:
                    fade_out = np.linspace(1, 0, fade_samples)
                    fade_in = np.linspace(0, 1, fade_samples)
                    
                    # 交叉淡化重叠部分
                    mixed_overlap = (overlap_part[-fade_samples:] * fade_out + 
                                   audio_float[:fade_samples] * fade_in)
                    
                    # 组合音频
                    processed_audio = np.concatenate([
                        overlap_part[:-fade_samples],
                        mixed_overlap,
                        audio_float[fade_samples:]
                    ])
                else:
                    processed_audio = np.concatenate([overlap_part, audio_float])
            else:
                processed_audio = np.concatenate([overlap_part, audio_float])

        # 使用简单下采样
        downsampled = self._simple_downsample(processed_audio)
        
        # 更新状态
        self.previous_chunk = audio_float

        # 简单的后处理
        clipped = np.clip(downsampled, -0.9, 0.9)
        
        # 转换为整数
        int16_audio = np.round(clipped * 32767.0).astype(np.int16).tobytes()
        ulaw_audio = audioop.lin2ulaw(int16_audio, 2)
        return base64.b64encode(ulaw_audio).decode("utf-8")

    def flush_base64_chunk(self) -> Optional[str]:
        if self.previous_chunk is not None and len(self.previous_chunk) > 0:
            # 处理最后一块
            final_downsampled = self._simple_downsample(self.previous_chunk)
            
            # 应用淡出
            fade_len = min(len(final_downsampled), 100)  # 约12ms的淡出
            if fade_len > 0:
                fade_out = np.linspace(1, 0, fade_len)
                final_downsampled[-fade_len:] *= fade_out
            
            clipped = np.clip(final_downsampled, -0.9, 0.9)
            int16_audio = np.round(clipped * 32767.0).astype(np.int16).tobytes()
            ulaw_audio = audioop.lin2ulaw(int16_audio, 2)
            
            self.previous_chunk = None
            return base64.b64encode(ulaw_audio).decode("utf-8")
        return None

    def reset(self):
        """重置处理器状态"""
        self.previous_chunk = None