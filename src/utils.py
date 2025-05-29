import audioop
import numpy as np
from scipy import signal

def ulaw_to_pcm24k(audio_bytes_ulaw, input_rate=8000, output_rate=24000):
    # μ-law → PCM 16-bit (8kHz)
    pcm_8k = audioop.ulaw2lin(audio_bytes_ulaw, 2)  # 2 bytes = 16-bit

    # 转成 numpy array
    audio_np = np.frombuffer(pcm_8k, dtype=np.int16)

    # 升采样到 24kHz
    num_samples = int(len(audio_np) * output_rate / input_rate)
    audio_resampled = signal.resample(audio_np, num_samples).astype(np.int16)

    return audio_resampled.tobytes()

def pcm24k_to_ulaw(pcm_data_24k: bytes, input_rate=24000, target_rate=8000) -> bytes:
    # Step 1: 转换为 numpy array，int16
    pcm_array = np.frombuffer(pcm_data_24k, dtype=np.int16)

    # Step 2: 降采样到 8000 Hz
    resample_len = int(len(pcm_array) * target_rate / input_rate)
    resampled = signal.resample(pcm_array, resample_len).astype(np.int16)

    # Step 3: 转换为 bytes
    resampled_bytes = resampled.tobytes()

    # Step 4: PCM -> μ-law
    ulaw_data = audioop.lin2ulaw(resampled_bytes, 2)  # 2 bytes per sample (16-bit)

    return ulaw_data