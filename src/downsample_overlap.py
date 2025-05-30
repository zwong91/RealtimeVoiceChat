import base64
import numpy as np
from scipy.signal import resample_poly
from typing import Optional
import audioop # For µ-law conversion

class DownsampleMuLawOverlap:
    SOURCE_RATE = 24000
    TARGET_RATE = 8000
    
    # Define overlap length in samples at the source rate (24kHz).
    # 50ms is a reasonable starting point.
    OVERLAP_SAMPLES_SOURCE = int(0.050 * SOURCE_RATE) # 1200 samples at 24kHz

    def __init__(self):
        # Initialize the overlap buffer with zeros.
        # This buffer stores the tail of the PREVIOUS 24kHz input chunk (or zeros for the first chunk).
        # It will be prepended to the current chunk before resampling.
        self.overlap_buffer_24kHz: np.ndarray = np.zeros(self.OVERLAP_SAMPLES_SOURCE, dtype=np.float32)
        # Keep track if any actual audio has been processed to refine flush behavior
        self.any_audio_processed: bool = False

    def get_base64_chunk(self, chunk: bytes) -> str:
        if not chunk:
            return ""

        audio_int16 = np.frombuffer(chunk, dtype=np.int16)
        if audio_int16.size == 0:
            return ""

        self.any_audio_processed = True # Mark that we've processed actual audio data

        # Convert 16-bit PCM to normalized float32
        current_audio_float = audio_int16.astype(np.float32) / 32768.0

        # Prepend the historical overlap data (or initial zeros) to the current audio chunk
        combined_input_24kHz = np.concatenate((self.overlap_buffer_24kHz, current_audio_float))

        # Resample the combined 24kHz data to 8kHz
        resampled_combined_8kHz = resample_poly(combined_input_24kHz, self.TARGET_RATE, self.SOURCE_RATE)

        # Calculate how many samples in the 8kHz output correspond to the prepended overlap buffer.
        # This is the portion to DISCARD from the beginning of the resampled output.
        # Since self.overlap_buffer_24kHz is initialized and maintained from tails, its length might vary
        # if input chunks are shorter than OVERLAP_SAMPLES_SOURCE.
        # However, our __init__ ensures it starts full length.
        # The original code's calculation is correct if `self.overlap_buffer_24kHz` can change length.
        # If we decide to always keep it `OVERLAP_SAMPLES_SOURCE` long (padded if necessary),
        # then `len(self.overlap_buffer_24kHz)` would always be `self.OVERLAP_SAMPLES_SOURCE`.
        # For this version, sticking to the original dynamic calculation based on actual buffer length:
        num_discard_samples_8kHz = round(len(self.overlap_buffer_24kHz) * self.TARGET_RATE / self.SOURCE_RATE)
        
        # Calculate the expected length of the current audio chunk after resampling to 8kHz.
        expected_output_len_current_chunk_8kHz = round(len(current_audio_float) * self.TARGET_RATE / self.SOURCE_RATE)

        # Extract the valid output segment:
        # Skip the resampled portion of the prepended overlap, take the portion for the current chunk.
        start_index = num_discard_samples_8kHz
        end_index = num_discard_samples_8kHz + expected_output_len_current_chunk_8kHz
        output_segment_float = resampled_combined_8kHz[start_index:end_index]
        
        # Handle edge case: if the calculated slice is empty but there's more data.
        # This can happen with very small final chunks or rounding.
        if len(output_segment_float) == 0 and len(resampled_combined_8kHz) > start_index:
             output_segment_float = resampled_combined_8kHz[start_index:]

        # Update `overlap_buffer_24kHz` for the NEXT call:
        # Save the TAIL of the CURRENT 24kHz input chunk.
        # If current chunk is shorter than OVERLAP_SAMPLES_SOURCE, the new overlap buffer will be shorter.
        self.overlap_buffer_24kHz = current_audio_float[-min(len(current_audio_float), self.OVERLAP_SAMPLES_SOURCE):]

        if output_segment_float.size == 0:
            return ""

        # Convert float32 8kHz audio to µ-law bytes and then Base64 encode
        pcm_int16_segment = (output_segment_float * 32767).astype(np.int16)
        # Ensure it's not empty bytes before µ-law conversion, which expects at least 1 sample (2 bytes for int16)
        if pcm_int16_segment.size == 0:
            return ""
        pcm_int16_bytes = pcm_int16_segment.tobytes()
        ulaw_bytes = audioop.lin2ulaw(pcm_int16_bytes, 2) # 2 for 2 bytes per sample (int16)
        
        return base64.b64encode(ulaw_bytes).decode('utf-8')

    def flush_base64_chunk(self) -> Optional[str]:
        # If no actual audio was processed, or the overlap buffer is empty, nothing to flush.
        if not self.any_audio_processed or self.overlap_buffer_24kHz is None or self.overlap_buffer_24kHz.size == 0:
            # Reset state for potential reuse
            self.overlap_buffer_24kHz = np.zeros(self.OVERLAP_SAMPLES_SOURCE, dtype=np.float32)
            self.any_audio_processed = False
            return None

        data_to_flush_24kHz = self.overlap_buffer_24kHz

        # For robust flushing of this tail segment, we should provide filter context.
        # Prepend leading zeros (simulating silence before this tail)
        # Append trailing zeros (allowing filter to ring out)
        # The length of these zero paddings can be OVERLAP_SAMPLES_SOURCE.
        leading_zeros = np.zeros(self.OVERLAP_SAMPLES_SOURCE, dtype=np.float32)
        trailing_zeros = np.zeros(self.OVERLAP_SAMPLES_SOURCE, dtype=np.float32) # Or shorter, e.g., filter_len

        # Create the input for resampling the final segment
        final_input_to_resample_24kHz = np.concatenate((leading_zeros, data_to_flush_24kHz, trailing_zeros))
        
        resampled_flushed_data_8kHz = resample_poly(final_input_to_resample_24kHz, self.TARGET_RATE, self.SOURCE_RATE)

        # Calculate indices to extract the part corresponding to data_to_flush_24kHz
        start_index_8k = round(len(leading_zeros) * self.TARGET_RATE / self.SOURCE_RATE)
        expected_len_data_8k = round(len(data_to_flush_24kHz) * self.TARGET_RATE / self.SOURCE_RATE)
        
        final_segment_float = resampled_flushed_data_8kHz[start_index_8k : start_index_8k + expected_len_data_8k]
        
        # Reset state for potential reuse
        self.overlap_buffer_24kHz = np.zeros(self.OVERLAP_SAMPLES_SOURCE, dtype=np.float32)
        self.any_audio_processed = False
        
        if final_segment_float.size == 0:
            return None

        # Convert to µ-law and Base64
        pcm_int16_segment = (final_segment_float * 32767).astype(np.int16)
        if pcm_int16_segment.size == 0:
            return None
        pcm_int16_bytes = pcm_int16_segment.tobytes()
        ulaw_bytes = audioop.lin2ulaw(pcm_int16_bytes, 2)
        
        return base64.b64encode(ulaw_bytes).decode('utf-8')