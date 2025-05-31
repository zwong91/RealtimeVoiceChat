import base64
import numpy as np
import audioop
from typing import Optional

class ResampleOverlap:
    """
    使用最简单可靠的重采样方法，完全避免复杂滤波造成的吱吱声。
    """

    def __init__(self, input_fs: int = 24000, output_fs: int = 8000, overlap_ms: int = 20):
        self.input_fs = input_fs
        self.output_fs = output_fs
        self.ratio = input_fs / output_fs  # 3.0
        
        self.overlap_samples_in = int(input_fs * overlap_ms / 1000)
        self.overlap_samples_out = int(output_fs * overlap_ms / 1000)
        
        self.previous_chunk = None
        self.leftover_samples = np.array([], dtype=np.float32)
        
        # 极简单的低通滤波系数，只是轻微平滑
        self.smooth_factor = 0.1

    def _simple_decimate(self, audio: np.ndarray) -> np.ndarray:
        """最简单的3:1下采样，先轻微平滑再抽取"""
        if len(audio) == 0:
            return audio
            
        # 超轻微的平滑，只是稍微减少高频
        smoothed = np.copy(audio)
        if len(audio) > 2:
            for i in range(1, len(audio) - 1):
                smoothed[i] = (1 - 2 * self.smooth_factor) * audio[i] + \
                             self.smooth_factor * (audio[i-1] + audio[i+1])
        
        # 直接每3个样本取1个
        decimated = smoothed[::3]
        return decimated

    def get_base64_chunk(self, chunk: bytes) -> str:
        if not chunk:
            return ""

        audio_int16 = np.frombuffer(chunk, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0

        # 拼接剩余样本和新样本
        if len(self.leftover_samples) > 0:
            combined_audio = np.concatenate([self.leftover_samples, audio_float])
        else:
            combined_audio = audio_float

        # 处理重叠
        if self.previous_chunk is not None:
            # 简单的线性交叉淡化
            overlap_len = min(self.overlap_samples_in, len(self.previous_chunk), len(combined_audio))
            if overlap_len > 0:
                fade_out = np.linspace(1, 0, overlap_len)
                fade_in = np.linspace(0, 1, overlap_len)
                
                overlap_region = (self.previous_chunk[-overlap_len:] * fade_out + 
                                combined_audio[:overlap_len] * fade_in)
                
                combined_audio = np.concatenate([
                    self.previous_chunk[:-overlap_len],
                    overlap_region,
                    combined_audio[overlap_len:]
                ])

        # 简单下采样
        downsampled = self._simple_decimate(combined_audio)
        
        # 计算需要保留多少样本给下次处理
        samples_used = len(downsampled) * 3
        if samples_used < len(combined_audio):
            self.leftover_samples = combined_audio[samples_used:]
        else:
            self.leftover_samples = np.array([], dtype=np.float32)

        # 更新上一块数据
        self.previous_chunk = audio_float

        # 最简单的后处理
        clipped = np.clip(downsampled, -0.95, 0.95)
        
        # 再次轻微平滑，减少量化噪声
        if len(clipped) > 2:
            final_smooth = np.copy(clipped)
            for i in range(1, len(clipped) - 1):
                final_smooth[i] = 0.7 * clipped[i] + 0.15 * (clipped[i-1] + clipped[i+1])
            clipped = final_smooth
        
        int16_audio = np.round(clipped * 32767.0).astype(np.int16).tobytes()
        ulaw_audio = audioop.lin2ulaw(int16_audio, 2)
        return base64.b64encode(ulaw_audio).decode("utf-8")

    def flush_base64_chunk(self) -> Optional[str]:
        if len(self.leftover_samples) > 0:
            # 处理剩余样本
            final_downsampled = self._simple_decimate(self.leftover_samples)
            
            if len(final_downsampled) > 0:
                # 应用淡出
                fade_len = min(len(final_downsampled), 20)
                if fade_len > 0:
                    fade_out = np.linspace(1, 0, fade_len)
                    final_downsampled[-fade_len:] *= fade_out
                
                clipped = np.clip(final_downsampled, -0.95, 0.95)
                
                # 同样的轻微平滑
                if len(clipped) > 2:
                    final_smooth = np.copy(clipped)
                    for i in range(1, len(clipped) - 1):
                        final_smooth[i] = 0.7 * clipped[i] + 0.15 * (clipped[i-1] + clipped[i+1])
                    clipped = final_smooth
                
                int16_audio = np.round(clipped * 32767.0).astype(np.int16).tobytes()
                ulaw_audio = audioop.lin2ulaw(int16_audio, 2)
                
                self.leftover_samples = np.array([], dtype=np.float32)
                return base64.b64encode(ulaw_audio).decode("utf-8")
        
        return None

    def reset(self):
        """重置处理器状态"""
        self.previous_chunk = None
        self.leftover_samples = np.array([], dtype=np.float32)