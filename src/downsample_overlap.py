import base64
import numpy as np
from scipy.signal import resample_poly
import audioop
from typing import Optional

class ResampleOverlap:
    """
    音频流按块重采样（24kHz 到 8kHz）并进行重叠处理。
    专门解决点击声和吱吱声问题。
    """

    def __init__(self, input_fs: int = 24000, output_fs: int = 8000, overlap_ms: int = 20):
        self.input_fs = input_fs
        self.output_fs = output_fs
        self.overlap_samples_in = int(input_fs * overlap_ms / 1000)
        self.previous_chunk = None
        self.previous_tail = None  # 保存上一块的尾部用于平滑连接
        self.initial_padding_samples_in = int(input_fs * overlap_ms / 1000)
        
        # 使用更保守的滤波参数
        self.kaiser_beta = 6  # 降低到6，减少振铃效应
        self.window = ('kaiser', self.kaiser_beta)
        
        # 用于平滑连接的渐变长度（很短，只处理边界）
        self.fade_samples = max(8, int(output_fs * 2 / 1000))  # 2ms的渐变

    def get_base64_chunk(self, chunk: bytes) -> str:
        if not chunk:
            return ""

        audio_int16 = np.frombuffer(chunk, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0

        if self.previous_chunk is None:
            # 第一块：添加零填充避免边界效应
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
            # 后续块：使用重叠处理
            combined = np.concatenate([
                self.previous_chunk[-self.overlap_samples_in:], 
                audio_float
            ])
            downsampled = resample_poly(
                combined,
                self.output_fs,
                self.input_fs,
                window=self.window
            )
            skip_samples = int(self.overlap_samples_in * self.output_fs / self.input_fs)
            clean_downsampled = downsampled[skip_samples:]

        # 关键改进：平滑连接处理
        if self.previous_tail is not None and len(clean_downsampled) > self.fade_samples:
            # 检查连接处是否有明显的跳跃
            connection_diff = clean_downsampled[0] - self.previous_tail[-1]
            
            # 如果跳跃幅度超过阈值，进行平滑处理
            if abs(connection_diff) > 0.01:  # 阈值可调
                # 创建短暂的渐变来消除跳跃
                fade_in = np.linspace(0, 1, self.fade_samples)
                correction = connection_diff * (1 - fade_in)
                clean_downsampled[:self.fade_samples] -= correction

        # 更新状态
        self.previous_chunk = audio_float
        
        # 保存当前块的尾部用于下次连接
        if len(clean_downsampled) > self.fade_samples:
            self.previous_tail = clean_downsampled[-self.fade_samples:]
        else:
            self.previous_tail = clean_downsampled

        # 最小化的后处理
        clipped = np.clip(clean_downsampled, -0.999, 0.999)  # 稍微保守的限幅
        
        # 转换为整数，使用舍入而不是截断
        int16_audio = np.round(clipped * 32767.0).astype(np.int16).tobytes()
        ulaw_audio = audioop.lin2ulaw(int16_audio, 2)
        return base64.b64encode(ulaw_audio).decode("utf-8")

    def flush_base64_chunk(self) -> Optional[str]:
        if self.previous_tail is not None and len(self.previous_tail) > 0:
            # 对最后一块应用轻微的淡出，避免突然结束
            final_chunk = np.copy(self.previous_tail)
            fade_out_len = min(len(final_chunk), self.fade_samples)
            if fade_out_len > 0:
                fade_out = np.linspace(1, 0, fade_out_len)
                final_chunk[-fade_out_len:] *= fade_out

            clipped = np.clip(final_chunk, -0.999, 0.999)
            int16_audio = np.round(clipped * 32767.0).astype(np.int16).tobytes()
            ulaw_audio = audioop.lin2ulaw(int16_audio, 2)
            
            # 清理状态
            self.previous_tail = None
            return base64.b64encode(ulaw_audio).decode("utf-8")
        return None

    def reset(self):
        """重置处理器状态"""
        self.previous_chunk = None
        self.previous_tail = None