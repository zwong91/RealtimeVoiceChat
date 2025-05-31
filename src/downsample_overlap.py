import base64
import numpy as np
from scipy.signal import resample_poly
from typing import Optional

def pcm_to_ulaw(pcm_data):
    """
    Convert PCM data to μ-law encoding.
    
    Args:
        pcm_data: numpy array of 16-bit PCM samples
    
    Returns:
        numpy array of μ-law encoded bytes
    """
    # Normalize to [-1, 1]
    normalized = pcm_data.astype(np.float32) / 32768.0
    
    # Apply μ-law compression (μ = 255)
    mu = 255.0
    sign = np.sign(normalized)
    magnitude = np.abs(normalized)
    
    # μ-law formula: sign * log(1 + μ * |x|) / log(1 + μ)
    compressed = sign * np.log1p(mu * magnitude) / np.log1p(mu)
    
    # Quantize to 8-bit (0-255)
    quantized = ((compressed + 1.0) * 127.5).astype(np.uint8)
    
    return quantized

class DownsampleOverlapUlaw:
    """
    Manages chunk-wise audio downsampling from 24kHz to 8kHz with 3x overlap handling and μ-law encoding.

    This class processes sequential audio chunks, downsamples them from 24kHz to 8kHz
    using `scipy.signal.resample_poly`, and manages 3x overlap between chunks to
    achieve maximum smoothness and eliminate boundary artifacts.
    """
    def __init__(self):
        """
        Initializes the DownsampleOverlapUlaw processor with 3x overlap strategy.
        """
        self.chunk_buffer = []  # Store recent chunks for 3x overlap
        self.output_buffer = []  # Store processed output segments

    def get_base64_chunk(self, chunk: bytes) -> str:
        """
        Processes an incoming audio chunk with 3x overlap strategy.

        For 3x overlap, we always process the current chunk together with the
        previous 2 chunks (when available), creating a sliding window that
        ensures maximum smoothness at all boundaries.

        Args:
            chunk: Raw audio data bytes (PCM 16-bit signed integer format expected).

        Returns:
            A Base64 encoded string representing the downsampled μ-law audio segment.
        """
        audio_int16 = np.frombuffer(chunk, dtype=np.int16)
        if audio_int16.size == 0:
            return ""

        audio_float = audio_int16.astype(np.float32) / 32768.0
        
        # Add current chunk to buffer
        self.chunk_buffer.append(audio_float)
        
        # Maintain only the last 3 chunks for 3x overlap
        if len(self.chunk_buffer) > 3:
            self.chunk_buffer.pop(0)
        
        # Always process with available chunks (1, 2, or 3 chunks)
        combined_audio = np.concatenate(self.chunk_buffer)
        
        # Downsample the combined audio
        downsampled = resample_poly(combined_audio, 8000, 24000)
        
        if len(self.chunk_buffer) == 1:
            # First chunk: return the entire downsampled chunk
            part = downsampled
        elif len(self.chunk_buffer) == 2:
            # Second chunk: return the second half of combined result
            # This gives us smooth transition from first to second chunk
            half_point = len(downsampled) // 2
            part = downsampled[half_point:]
        else:
            # Third chunk and beyond: return the middle third
            # This is the core of 3x overlap - always extract the middle portion
            # which benefits from smoothing on both sides
            third_len = len(downsampled) // 3
            start_idx = third_len
            end_idx = 2 * third_len
            part = downsampled[start_idx:end_idx]

        # Convert to μ-law
        pcm_16bit = (part * 32767).astype(np.int16)
        ulaw_data = pcm_to_ulaw(pcm_16bit)
        
        return base64.b64encode(ulaw_data.tobytes()).decode('utf-8')

    def flush_base64_chunk(self) -> Optional[str]:
        """
        Returns the final remaining segments after all chunks are processed.
        
        For 3x overlap, we need to flush the remaining portions that haven't
        been output yet - typically the last third of the final combined processing.
        
        Returns:
            A Base64 encoded string containing the final audio segments,
            or None if no chunks were processed.
        """
        if not self.chunk_buffer:
            return None
            
        # Process the final state
        if len(self.chunk_buffer) >= 2:
            # We have at least 2 chunks, so we can extract the final portion
            combined_audio = np.concatenate(self.chunk_buffer)
            downsampled = resample_poly(combined_audio, 8000, 24000)
            
            # Return the final third (the part we haven't output yet)
            third_len = len(downsampled) // 3
            if len(self.chunk_buffer) == 2:
                # For 2 chunks, we've output the second half, so no remaining portion
                remaining_part = np.array([])
            else:
                # For 3+ chunks, return the final third
                start_idx = 2 * third_len
                remaining_part = downsampled[start_idx:]
            
            if len(remaining_part) > 0:
                pcm_16bit = (remaining_part * 32767).astype(np.int16)
                ulaw_data = pcm_to_ulaw(pcm_16bit)
                
                # Clear state
                self.chunk_buffer = []
                return base64.b64encode(ulaw_data.tobytes()).decode('utf-8')
        
        # Clear state
        self.chunk_buffer = []
        return None


# 使用示例和测试
def example_usage():
    """
    Example usage demonstrating 3x overlap processing
    """
    processor = DownsampleOverlapUlaw()
    
    # 模拟24kHz PCM数据
    sample_rate = 24000
    chunk_duration = 0.1  # 100ms chunks
    samples_per_chunk = int(sample_rate * chunk_duration)
    
    # 生成测试信号 - 多频率混合更好测试平滑性
    t_total = np.linspace(0, 1.0, sample_rate)  # 1秒总时长
    test_signal = (
        np.sin(2 * np.pi * 440 * t_total) * 0.3 +  # 440Hz
        np.sin(2 * np.pi * 880 * t_total) * 0.2 +  # 880Hz 
        np.sin(2 * np.pi * 220 * t_total) * 0.1    # 220Hz
    )
    test_signal = (test_signal * 16384).astype(np.int16)
    
    # 分割为块
    chunks = []
    for i in range(0, len(test_signal), samples_per_chunk):
        chunk_data = test_signal[i:i+samples_per_chunk]
        if len(chunk_data) == samples_per_chunk:  # 只处理完整块
            chunks.append(chunk_data.tobytes())
    
    print(f"处理 {len(chunks)} 个音频块，每块 {chunk_duration*1000}ms")
    print("使用3倍重叠策略进行24kHz→8kHz降采样 + μ-law编码")
    print("-" * 50)
    
    # 处理每个块
    results = []
    for i, chunk in enumerate(chunks):
        result = processor.get_base64_chunk(chunk)
        results.append(result)
        
        # 解码验证大小
        if result:
            decoded_size = len(base64.b64decode(result))
            print(f"块 {i+1:2d}: {len(result):4d} chars -> {decoded_size:3d} μ-law bytes")
        else:
            print(f"块 {i+1:2d}: 空输出")
    
    # 获取最终剩余部分
    final_chunk = processor.flush_base64_chunk()
    if final_chunk:
        final_size = len(base64.b64decode(final_chunk))
        results.append(final_chunk)
        print(f"最终块: {len(final_chunk):4d} chars -> {final_size:3d} μ-law bytes")
    
    print(f"\n总共输出 {len([r for r in results if r])} 个有效音频段")
    return results

if __name__ == "__main__":
    example_usage()