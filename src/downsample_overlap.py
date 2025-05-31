import base64
import numpy as np
from scipy.signal import resample_poly, get_window
import audioop
from typing import Optional

class ResampleOverlap:
    """
    改进的音频流按块重采样（24kHz 到 8kHz）并进行重叠处理。
    专门优化以消除刺耳的机器音效。
    """

    def __init__(self, input_fs: int = 24000, output_fs: int = 8000, overlap_ms: int = 16):
        self.input_fs = input_fs
        self.output_fs = output_fs
        
        # 增加重叠时间以获得更平滑的过渡
        self.overlap_samples_in = int(input_fs * overlap_ms / 1000)
        self.overlap_samples_out = int(output_fs * overlap_ms / 1000)
        
        # 状态变量
        self.previous_chunk = None
        self.overlap_buffer = None
        
        # 初始填充
        self.initial_padding_samples_in = int(input_fs * overlap_ms / 1000)
        
        # 优化的滤波器参数 - 更强的抗混叠
        self.kaiser_beta = 12  # 增加到12获得更好的抗混叠效果
        
        # 创建渐变窗口用于平滑重叠区域
        self.fade_samples = self.overlap_samples_out
        self.fade_in = np.linspace(0, 1, self.fade_samples)
        self.fade_out = np.linspace(1, 0, self.fade_samples)
        
        # 高通滤波器系数用于去除低频噪声
        self.highpass_alpha = 0.95

    def _apply_highpass_filter(self, audio: np.ndarray) -> np.ndarray:
        """应用简单的高通滤波器去除低频噪声"""
        if len(audio) == 0:
            return audio
            
        filtered = np.zeros_like(audio)
        filtered[0] = audio[0]
        
        for i in range(1, len(audio)):
            filtered[i] = self.highpass_alpha * (filtered[i-1] + audio[i] - audio[i-1])
            
        return filtered

    def _smooth_overlap(self, new_chunk: np.ndarray) -> np.ndarray:
        """使用渐变窗口平滑重叠区域"""
        if self.overlap_buffer is None:
            return new_chunk
            
        if len(new_chunk) < self.fade_samples:
            # 如果新块太短，直接返回
            return new_chunk
            
        # 创建重叠区域
        overlap_region = new_chunk[:self.fade_samples]
        remaining_region = new_chunk[self.fade_samples:]
        
        # 应用交叉渐变
        if len(self.overlap_buffer) >= self.fade_samples:
            overlap_out = self.overlap_buffer[-self.fade_samples:] * self.fade_out
            overlap_in = overlap_region * self.fade_in
            smooth_overlap = overlap_out + overlap_in
            
            # 组合结果
            result = np.concatenate([
                self.overlap_buffer[:-self.fade_samples],
                smooth_overlap,
                remaining_region
            ])
        else:
            # 如果重叠缓冲区太短，简单拼接
            result = np.concatenate([self.overlap_buffer, new_chunk])
            
        return result

    def get_base64_chunk(self, chunk: bytes) -> str:
        if not chunk:
            return ""

        # 转换为浮点数组
        audio_int16 = np.frombuffer(chunk, dtype=np.int16)
        audio_float = audio_int16.astype(np.float64) / 32768.0  # 使用更高精度

        # 第一次处理
        if self.previous_chunk is None:
            # 添加初始填充以避免边界效应
            padded_audio_float = np.concatenate([
                np.zeros(self.initial_padding_samples_in, dtype=np.float64),
                audio_float
            ])
            
            # 使用更高质量的重采样
            downsampled = resample_poly(
                padded_audio_float,
                self.output_fs,
                self.input_fs,
                window=('kaiser', self.kaiser_beta),
                padtype='constant'  # 使用常量填充
            )
            
            # 移除初始填充
            padding_samples_out = int(self.initial_padding_samples_in * self.output_fs / self.input_fs)
            clean_downsampled = downsampled[padding_samples_out:]
            
        else:
            # 创建重叠的组合信号
            combined = np.concatenate([
                self.previous_chunk[-self.overlap_samples_in:], 
                audio_float
            ])
            
            # 高质量重采样
            downsampled = resample_poly(
                combined,
                self.output_fs,
                self.input_fs,
                window=('kaiser', self.kaiser_beta),
                padtype='constant'
            )
            
            # 移除重叠部分
            skip_samples = int(self.overlap_samples_in * self.output_fs / self.input_fs)
            clean_downsampled = downsampled[skip_samples:]

        # 应用高通滤波器去除低频噪声
        clean_downsampled = self._apply_highpass_filter(clean_downsampled)
        
        # 应用平滑重叠处理
        if self.overlap_buffer is not None:
            smooth_result = self._smooth_overlap(clean_downsampled)
            # 保留部分用于下次重叠
            if len(smooth_result) > self.overlap_samples_out:
                output_chunk = smooth_result[:-self.overlap_samples_out]
                self.overlap_buffer = smooth_result[-self.overlap_samples_out:]
            else:
                output_chunk = smooth_result
                self.overlap_buffer = None
        else:
            if len(clean_downsampled) > self.overlap_samples_out:
                output_chunk = clean_downsampled[:-self.overlap_samples_out]
                self.overlap_buffer = clean_downsampled[-self.overlap_samples_out:]
            else:
                output_chunk = clean_downsampled
                self.overlap_buffer = None

        # 更新状态
        self.previous_chunk = audio_float

        # 音频后处理
        # 1. 限制幅度防止削波
        clipped = np.clip(output_chunk, -0.98, 0.98)  # 留一点余量
        
        # 2. 轻微的噪声门限，去除极小的信号
        noise_floor = 0.001
        clipped = np.where(np.abs(clipped) < noise_floor, 0, clipped)
        
        # 3. 转换为16位整数
        int16_audio = (clipped * 32767.0).astype(np.int16)
        
        # 4. 应用轻微的平滑处理
        if len(int16_audio) > 1:
            # 简单的移动平均平滑
            smoothed = np.copy(int16_audio).astype(np.float32)
            for i in range(1, len(smoothed)-1):
                smoothed[i] = 0.25 * smoothed[i-1] + 0.5 * smoothed[i] + 0.25 * smoothed[i+1]
            int16_audio = smoothed.astype(np.int16)
        
        # 转换为字节并编码
        int16_bytes = int16_audio.tobytes()
        ulaw_audio = audioop.lin2ulaw(int16_bytes, 2)
        return base64.b64encode(ulaw_audio).decode("utf-8")

    def flush_base64_chunk(self) -> Optional[str]:
        """刷新最后的重叠缓冲区"""
        if self.overlap_buffer is not None and len(self.overlap_buffer) > 0:
            # 应用轻微的淡出以避免突然结束
            fade_out_samples = min(len(self.overlap_buffer), self.fade_samples // 2)
            if fade_out_samples > 0:
                fade_out_window = np.linspace(1, 0, fade_out_samples)
                self.overlap_buffer[-fade_out_samples:] *= fade_out_window
            
            # 应用相同的后处理
            clipped = np.clip(self.overlap_buffer, -0.98, 0.98)
            noise_floor = 0.001
            clipped = np.where(np.abs(clipped) < noise_floor, 0, clipped)
            
            int16_audio = (clipped * 32767.0).astype(np.int16)
            
            # 平滑处理
            if len(int16_audio) > 1:
                smoothed = np.copy(int16_audio).astype(np.float32)
                for i in range(1, len(smoothed)-1):
                    smoothed[i] = 0.25 * smoothed[i-1] + 0.5 * smoothed[i] + 0.25 * smoothed[i+1]
                int16_audio = smoothed.astype(np.int16)
            
            int16_bytes = int16_audio.tobytes()
            ulaw_audio = audioop.lin2ulaw(int16_bytes, 2)
            
            # 清理状态
            self.overlap_buffer = None
            return base64.b64encode(ulaw_audio).decode("utf-8")
        
        return None

    def reset(self):
        """重置处理器状态"""
        self.previous_chunk = None
        self.overlap_buffer = None