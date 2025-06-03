import base64
import numpy as np
from scipy.signal import resample_poly, windows, lfilter, butter
import audioop
from typing import Optional

class ResampleOverlap:
    """
    音频流按块重采样（24kHz 到 8kHz）并进行重叠处理。
    专门解决点击声和吱吱声问题。

    改进点：
    - 更精细的重采样窗口选择和参数。
    - 改进的平滑连接逻辑，增加可配置性。
    - 可选的低通滤波以进一步抑制高频噪音。
    - 更好的状态管理和注释。
    """

    def __init__(self, input_fs: int = 24000, output_fs: int = 8000,
                 overlap_ms: int = 20, # 增加重叠时间，提供更长的平滑区域
                 fade_ms: int = 5, # 渐变长度增加，提供更平滑的衔接
                 apply_lowpass: bool = True, # 增加一个选项来控制是否应用低通滤波
                 lowpass_cutoff_hz: int = 3800, # 针对8kHz采样率，截止频率应低于奈奎斯特频率（4kHz）
                 lowpass_order: int = 4 # 滤波器阶数，越高越陡峭，但可能引入更多延迟
                 ):
        self.input_fs = input_fs
        self.output_fs = output_fs
        self.overlap_samples_in = int(input_fs * overlap_ms / 1000)
        
        # 确保重叠至少为一个重采样周期，或者足够大
        # 考虑到 resample_poly 会有一定的延迟，较大的重叠有助于更好的衔接
        if self.overlap_samples_in == 0:
            self.overlap_samples_in = int(input_fs * 5 / 1000) # 至少5ms

        self.previous_input_tail = None # 保存上一块原始采样率音频的尾部
        self.previous_output_tail = None # 保存上一块重采样后音频的尾部，用于平滑连接

        # 初始零填充，用于第一块的平滑启动
        self.initial_padding_samples_in = self.overlap_samples_in * 2 # 增加初始填充量

        # 重采样窗口：'kaiser' 是一种不错的选择，可以调整 beta 值
        # beta 值越大，旁瓣衰减越大，主瓣越宽（过渡带越宽）
        # 尝试使用不同的beta值，通常在 5 到 14 之间
        self.kaiser_beta = 12 # 稍微增加 beta，进一步抑制旁瓣，减少高频噪音
        self.window = ('kaiser', self.kaiser_beta)
        
        # 用于平滑连接的渐变长度
        self.fade_samples = max(8, int(output_fs * fade_ms / 1000))

        # 低通滤波器参数
        self.apply_lowpass = apply_lowpass
        if self.apply_lowpass:
            # 设计 Butterworth 滤波器
            nyquist = 0.5 * self.output_fs
            normal_cutoff = lowpass_cutoff_hz / nyquist
            self.b, self.a = butter(lowpass_order, normal_cutoff, btype='low', analog=False)
            self.filter_zi = None # 滤波器状态，用于跨块保持连续性

        # 确保数据类型一致性，避免不必要的转换
        self.dtype_float = np.float32

        # 跟踪是否是第一个块
        self._is_first_chunk = True

    def _apply_fade(self, signal: np.ndarray, fade_len: int, fade_type: str = 'linear') -> np.ndarray:
        """
        对信号应用淡入或淡出。
        fade_type: 'linear', 'hanning'
        """
        if fade_len <= 0 or len(signal) < fade_len:
            return signal

        if fade_type == 'linear':
            fade_curve_in = np.linspace(0, 1, fade_len)
            fade_curve_out = np.linspace(1, 0, fade_len)
        elif fade_type == 'hanning':
            fade_curve_in = windows.hanning(fade_len * 2)[fade_len:] # 半个 Hanning 窗作为淡入
            fade_curve_out = windows.hanning(fade_len * 2)[:fade_len] # 半个 Hanning 窗作为淡出
        else:
            raise ValueError(f"Unknown fade type: {fade_type}")

        out_signal = np.copy(signal)
        out_signal[:fade_len] *= fade_curve_in[:len(out_signal[:fade_len])]
        out_signal[-fade_len:] *= fade_curve_out[-len(out_signal[-fade_len:]):]
        return out_signal

    def get_base64_chunk(self, chunk: bytes) -> str:
        if not chunk:
            return ""

        audio_int16 = np.frombuffer(chunk, dtype=np.int16)
        audio_float = audio_int16.astype(self.dtype_float) / 32768.0

        processed_audio_float = None

        if self._is_first_chunk:
            # 第一块：添加零填充以避免边界效应，并应用淡入
            padded_audio_float = np.concatenate([
                np.zeros(self.initial_padding_samples_in, dtype=self.dtype_float),
                audio_float
            ])
            
            # 执行重采样
            downsampled = resample_poly(
                padded_audio_float,
                self.output_fs,
                self.input_fs,
                window=self.window
            ).astype(self.dtype_float) # 确保数据类型一致

            padding_samples_out = int(self.initial_padding_samples_in * self.output_fs / self.input_fs)
            processed_audio_float = downsampled[padding_samples_out:]
            self._is_first_chunk = False

        else:
            # 后续块：使用重叠处理
            # 拼接上一块的尾部和当前块，进行重采样
            # previous_input_tail 存储的是原始采样率的尾部
            if self.previous_input_tail is not None:
                combined_input = np.concatenate([
                    self.previous_input_tail,
                    audio_float
                ])
            else:
                # 理论上不会发生，除非 reset 后没有正确处理
                combined_input = audio_float 

            downsampled_combined = resample_poly(
                combined_input,
                self.output_fs,
                self.input_fs,
                window=self.window
            ).astype(self.dtype_float)

            # 计算应该跳过多少重叠样本
            # 这里需要精确计算重叠部分在输出中的长度
            skip_samples_out = int(len(self.previous_input_tail) * self.output_fs / self.input_fs) if self.previous_input_tail is not None else 0

            # 提取当前块的有效输出部分
            current_chunk_output = downsampled_combined[skip_samples_out:]

            # 平滑连接处理
            if self.previous_output_tail is not None and len(current_chunk_output) > 0:
                # 找到重叠区域在当前块的起始点
                overlap_start_idx = 0
                
                # 计算需要重叠的长度，取当前块输出的长度和渐变长度的最小值
                effective_fade_len = min(self.fade_samples, len(self.previous_output_tail), len(current_chunk_output))

                if effective_fade_len > 0:
                    # 创建两个渐变曲线
                    fade_out_curve = np.linspace(1, 0, effective_fade_len)
                    fade_in_curve = np.linspace(0, 1, effective_fade_len)

                    # 对前一个块的尾部应用淡出，对当前块的开头应用淡入
                    # 这样做是为了消除两个块连接处的可能不连续性
                    # 将两个重叠部分叠加，实现平滑混合
                    overlap_blend = (self.previous_output_tail[-effective_fade_len:] * fade_out_curve) + \
                                    (current_chunk_output[overlap_start_idx : overlap_start_idx + effective_fade_len] * fade_in_curve)
                    
                    # 将混合后的部分放回当前块的开头
                    current_chunk_output[overlap_start_idx : overlap_start_idx + effective_fade_len] = overlap_blend

            processed_audio_float = current_chunk_output

        # 更新状态：保存当前原始采样率块的尾部和重采样后块的尾部
        self.previous_input_tail = audio_float[-self.overlap_samples_in:] if len(audio_float) >= self.overlap_samples_in else audio_float
        
        # 确保 previous_output_tail 至少有 fade_samples 长度
        if len(processed_audio_float) >= self.fade_samples:
            self.previous_output_tail = processed_audio_float[-self.fade_samples:]
        else:
            self.previous_output_tail = processed_audio_float # 如果不够长，就保存全部

        # 可选的后期低通滤波
        if self.apply_lowpass and len(processed_audio_float) > 0:
            if self.filter_zi is None:
                # 第一次应用滤波器，初始化状态
                processed_audio_float, self.filter_zi = lfilter(self.b, self.a, processed_audio_float, zi=np.zeros((max(len(self.b), len(self.a)) - 1,)))
            else:
                # 延续前一个块的滤波器状态
                processed_audio_float, self.filter_zi = lfilter(self.b, self.a, processed_audio_float, zi=self.filter_zi)

        # 最小化的后处理：限幅
        clipped = np.clip(processed_audio_float, -0.999, 0.999) # 稍微保守的限幅，避免削波失真

        # 转换为整数，使用舍入而不是截断
        int16_audio = np.round(clipped * 32767.0).astype(np.int16).tobytes()
        ulaw_audio = audioop.lin2ulaw(int16_audio, 2)
        return base64.b64encode(ulaw_audio).decode("utf-8")

    def flush_base64_chunk(self) -> Optional[str]:
        """
        处理剩余的尾部数据，并应用淡出效果以平滑结束。
        """
        if self.previous_output_tail is not None and len(self.previous_output_tail) > 0:
            final_chunk = np.copy(self.previous_output_tail)
            
            # 对最后一块应用轻微的淡出
            fade_out_len = min(len(final_chunk), self.fade_samples * 2) # 最后的淡出可以稍微长一点
            if fade_out_len > 0:
                fade_out = np.linspace(1, 0, fade_out_len)
                final_chunk[-fade_out_len:] *= fade_out

            clipped = np.clip(final_chunk, -0.999, 0.999)
            int16_audio = np.round(clipped * 32767.0).astype(np.int16).tobytes()
            ulaw_audio = audioop.lin2ulaw(int16_audio, 2)
            
            # 清理状态
            self.reset() # 调用 reset 来完全清理状态
            return base64.b64encode(ulaw_audio).decode("utf-8")
        return None

    def reset(self):
        """重置处理器状态"""
        self.previous_input_tail = None
        self.previous_output_tail = None
        self._is_first_chunk = True
        self.filter_zi = None # 重置滤波器状态