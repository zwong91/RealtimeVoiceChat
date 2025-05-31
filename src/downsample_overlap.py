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
    Manages chunk-wise audio downsampling from 24kHz to 8kHz with overlap handling and μ-law encoding.

    This class processes sequential audio chunks, downsamples them from 24kHz to 8kHz
    using `scipy.signal.resample_poly`, manages overlap between chunks to
    mitigate boundary artifacts, and converts to μ-law encoding. The processed,
    downsampled audio segments are returned as Base64 encoded strings.
    """
    def __init__(self):
        """
        Initializes the DownsampleOverlapUlaw processor.

        Sets up the internal state required for tracking previous audio chunks
        and their resampled versions to handle overlaps during processing.
        """
        self.previous_chunk: Optional[np.ndarray] = None
        self.resampled_previous_chunk: Optional[np.ndarray] = None

    def get_base64_chunk(self, chunk: bytes) -> str:
        """
        Processes an incoming audio chunk, downsamples it to 8kHz, applies μ-law encoding, 
        and returns the relevant segment as Base64.

        Converts the raw PCM bytes (assumed 16-bit signed integer) chunk to a
        float32 numpy array, normalizes it, and downsamples from 24kHz to 8kHz.
        It uses the previous chunk's data to create an overlap, resamples the
        combined audio, and extracts the central portion corresponding primarily
        to the current chunk, using overlap to smooth transitions. The extracted
        audio segment is converted to μ-law encoding and returned as Base64.

        Args:
            chunk: Raw audio data bytes (PCM 16-bit signed integer format expected).

        Returns:
            A Base64 encoded string representing the downsampled μ-law audio segment
            corresponding to the input chunk, adjusted for overlap. Returns an
            empty string if the input chunk is empty.
        """
        audio_int16 = np.frombuffer(chunk, dtype=np.int16)
        # Handle potential empty chunks gracefully
        if audio_int16.size == 0:
            return ""  # Return empty string for empty input chunk

        audio_float = audio_int16.astype(np.float32) / 32768.0

        # Downsample the current chunk independently first
        downsampled_current_chunk = resample_poly(audio_float, 8000, 24000)

        if self.previous_chunk is None:
            # First chunk: Use overlap strategy adapted for downsampling
            # Since we're going from 24kHz to 8kHz (1:3 ratio), we use more aggressive overlap
            overlap_size = len(downsampled_current_chunk) * 2 // 3  # Use 2/3 instead of 1/2
            part = downsampled_current_chunk[:overlap_size]
        else:
            # Subsequent chunks: Combine previous float chunk with current float chunk
            combined = np.concatenate((self.previous_chunk, audio_float))
            # Downsample the combined chunk
            down = resample_poly(combined, 8000, 24000)

            # Calculate lengths and indices for extracting the middle part
            assert self.resampled_previous_chunk is not None
            prev_len = len(self.resampled_previous_chunk)  # Length of the downsampled previous chunk
            
            # Adjusted overlap calculation for downsampling
            # Use 1/3 overlap point from previous chunk (more conservative than //2)
            h_prev = prev_len * 2 // 3  # Start from 2/3 point of previous chunk
            
            # Calculate the end index for the current chunk's contribution
            # More aggressive overlap for downsampling to ensure smoothness
            current_contribution_len = len(down) - prev_len
            h_cur = prev_len + current_contribution_len * 2 // 3

            part = down[h_prev:h_cur]

        # Update state for the next iteration
        self.previous_chunk = audio_float
        self.resampled_previous_chunk = downsampled_current_chunk

        # Convert to 16-bit PCM first, then to μ-law
        pcm_16bit = (part * 32767).astype(np.int16)
        ulaw_data = pcm_to_ulaw(pcm_16bit)
        
        return base64.b64encode(ulaw_data.tobytes()).decode('utf-8')

    def flush_base64_chunk(self) -> Optional[str]:
        """
        Returns the final remaining segment of downsampled μ-law audio after all chunks are processed.

        After the last call to `get_base64_chunk`, this method returns the remaining
        portion of the final upsampled chunk, converted to μ-law encoding and 
        encoded as Base64. It then clears the internal state.

        Returns:
            A Base64 encoded string containing the final downsampled μ-law audio chunk,
            or None if no chunks were processed or if flush has already been called.
        """
        if self.resampled_previous_chunk is not None:
            # Return the remaining portion of the last downsampled chunk
            remaining_start = len(self.resampled_previous_chunk) * 2 // 3
            remaining_part = self.resampled_previous_chunk[remaining_start:]
            
            # Convert to μ-law
            pcm_16bit = (remaining_part * 32767).astype(np.int16)
            ulaw_data = pcm_to_ulaw(pcm_16bit)

            # Clear state after flushing
            self.previous_chunk = None
            self.resampled_previous_chunk = None
            
            return base64.b64encode(ulaw_data.tobytes()).decode('utf-8')
        return None


# 使用示例
def example_usage():
    """
    Example usage of the DownsampleOverlapUlaw class
    """
    processor = DownsampleOverlapUlaw()
    
    # 模拟一些24kHz的PCM数据块
    sample_rate = 24000
    chunk_duration = 0.1  # 100ms chunks
    samples_per_chunk = int(sample_rate * chunk_duration)
    
    # 生成测试数据 (模拟语音信号)
    t = np.linspace(0, chunk_duration * 3, samples_per_chunk * 3)
    test_signal = (np.sin(2 * np.pi * 440 * t) * 16384).astype(np.int16)  # 440Hz sine wave
    
    # 分割为块
    chunks = [test_signal[i:i+samples_per_chunk].tobytes() 
              for i in range(0, len(test_signal), samples_per_chunk)]
    
    # 处理每个块
    results = []
    for i, chunk in enumerate(chunks):
        result = processor.get_base64_chunk(chunk)
        results.append(result)
        print(f"Chunk {i+1}: {len(result)} characters")
    
    # 获取最后的剩余部分
    final_chunk = processor.flush_base64_chunk()
    if final_chunk:
        results.append(final_chunk)
        print(f"Final chunk: {len(final_chunk)} characters")
    
    return results

if __name__ == "__main__":
    example_usage()