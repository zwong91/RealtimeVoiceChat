import base64
import numpy as np
from scipy.signal import resample_poly
from typing import Optional
import audioop # For µ-law conversion
import math # For math.ceil, though resample_poly uses it internally

class DownsampleMuLawOverlap:
    SOURCE_RATE = 24000
    TARGET_RATE = 8000
    
    # Define overlap length in samples at the source rate (24kHz).
    # This should be generous enough to cover FIR filter transients. 50ms is common.
    OVERLAP_SAMPLES_SOURCE = int(0.050 * SOURCE_RATE) # 1200 samples

    # Pre-calculate the number of samples in the target rate that correspond to the overlap.
    # This is the fixed amount to discard from the beginning of each resampled block.
    SAMPLES_TO_DISCARD_TARGET_RATE = int(round(OVERLAP_SAMPLES_SOURCE * TARGET_RATE / SOURCE_RATE)) # 1200 * (8/24) = 400

    def __init__(self):
        # This buffer stores the tail of the logical continuous input stream,
        # specifically, the last OVERLAP_SAMPLES_SOURCE samples.
        # It is always OVERLAP_SAMPLES_SOURCE long and prepended to the current chunk.
        self.overlap_state_24kHz: np.ndarray = np.zeros(self.OVERLAP_SAMPLES_SOURCE, dtype=np.float32)
        self.any_audio_processed: bool = False # Tracks if any non-empty chunk has been processed
        self._reset_state() # Initialize correctly

    def _reset_state(self):
        self.overlap_state_24kHz = np.zeros(self.OVERLAP_SAMPLES_SOURCE, dtype=np.float32)
        self.any_audio_processed = False

    def get_base64_chunk(self, chunk: bytes) -> str:
        # Convert chunk to float32 audio data
        if not chunk:
            # An empty chunk might be meaningful if audio was already processed (e.g., pause or end of segment)
            # If no audio ever, then it's a true no-op.
            if not self.any_audio_processed:
                return ""
            current_audio_float = np.array([], dtype=np.float32)
        else:
            audio_int16 = np.frombuffer(chunk, dtype=np.int16)
            if audio_int16.size == 0:
                if not self.any_audio_processed:
                    return ""
                current_audio_float = np.array([], dtype=np.float32)
            else:
                # Only mark as processed if actual data samples are present
                if audio_int16.size > 0:
                    self.any_audio_processed = True
                current_audio_float = audio_int16.astype(np.float32) / 32768.0

        # Concatenate the preserved overlap state with the new audio data
        block_to_resample_24kHz = np.concatenate((self.overlap_state_24kHz, current_audio_float))

        # Resample the combined block
        resampled_block_8kHz = resample_poly(block_to_resample_24kHz, self.TARGET_RATE, self.SOURCE_RATE)

        # Determine the segment of resampled_block_8kHz to keep.
        # We discard the part corresponding to self.overlap_state_24kHz.
        start_idx = self.SAMPLES_TO_DISCARD_TARGET_RATE
        
        # The rest of the resampled block corresponds to current_audio_float.
        # resample_poly's output length is math.ceil(len(input) * ratio).
        # By taking [start_idx:], we get all samples from that point onwards.
        if start_idx < len(resampled_block_8kHz):
            output_segment_float = resampled_block_8kHz[start_idx:]
        else:
            # This can happen if block_to_resample_24kHz was so short (e.g. only initial overlap)
            # that after resampling, its length is <= SAMPLES_TO_DISCARD_TARGET_RATE.
            # Or if current_audio_float was empty.
            output_segment_float = np.array([], dtype=np.float32) 

        # Update self.overlap_state_24kHz for the *next* iteration.
        # It must be OVERLAP_SAMPLES_SOURCE long.
        # It should contain the tail of current_audio_float, padded with the
        # end of the *previous* overlap_state_24kHz if current_audio_float is too short.
        if len(current_audio_float) >= self.OVERLAP_SAMPLES_SOURCE:
            self.overlap_state_24kHz = current_audio_float[-self.OVERLAP_SAMPLES_SOURCE:]
        else:
            # current_audio_float is shorter than the overlap length.
            # The new state is [ส่วนท้ายของสถานะเก่า | current_audio_float ทั้งหมด]
            num_needed_from_old_overlap = self.OVERLAP_SAMPLES_SOURCE - len(current_audio_float)
            self.overlap_state_24kHz = np.concatenate((self.overlap_state_24kHz[-num_needed_from_old_overlap:], current_audio_float))
        
        if output_segment_float.size == 0:
            return ""

        # Convert to µ-law
        pcm_int16_segment = (output_segment_float * 32767).astype(np.int16)
        # audioop.lin2ulaw expects at least one sample (2 bytes for int16)
        if pcm_int16_segment.size == 0:
            return ""
        pcm_int16_bytes = pcm_int16_segment.tobytes()
        ulaw_bytes = audioop.lin2ulaw(pcm_int16_bytes, 2) # 2 bytes per int16 sample
        
        return base64.b64encode(ulaw_bytes).decode('utf-8')

    def flush_base64_chunk(self) -> Optional[str]:
        # If no actual audio data was ever processed, nothing to flush.
        if not self.any_audio_processed:
            self._reset_state() # Ensure clean state for potential reuse
            return None
        
        # The self.overlap_state_24kHz contains the last OVERLAP_SAMPLES_SOURCE
        # samples of the input stream. These need to be processed and output.
        data_to_flush_24kHz = self.overlap_state_24kHz

        # To resample this final segment cleanly, pad it with zeros on both sides
        # for filter settling (leading_zeros) and ring-out (trailing_zeros).
        leading_zeros = np.zeros(self.OVERLAP_SAMPLES_SOURCE, dtype=np.float32)
        trailing_zeros = np.zeros(self.OVERLAP_SAMPLES_SOURCE, dtype=np.float32) 

        final_input_to_resample_24kHz = np.concatenate((leading_zeros, data_to_flush_24kHz, trailing_zeros))
        
        resampled_flushed_data_8kHz = resample_poly(final_input_to_resample_24kHz, self.TARGET_RATE, self.SOURCE_RATE)

        # Extract the portion corresponding to data_to_flush_24kHz.
        # Start index skips the resampled leading_zeros.
        start_idx_8k = self.SAMPLES_TO_DISCARD_TARGET_RATE # Corresponds to len(leading_zeros) resampled
        
        # The length of the actual data we want (resampled data_to_flush_24kHz).
        # Since data_to_flush_24kHz is OVERLAP_SAMPLES_SOURCE long, its resampled length is SAMPLES_TO_DISCARD_TARGET_RATE.
        len_of_flushed_segment_8k = self.SAMPLES_TO_DISCARD_TARGET_RATE
        end_idx_8k = start_idx_8k + len_of_flushed_segment_8k
        
        # Slice carefully to get only the part corresponding to data_to_flush_24kHz
        # and not the resampled trailing_zeros.
        if start_idx_8k < len(resampled_flushed_data_8kHz):
            final_segment_float = resampled_flushed_data_8kHz[start_idx_8k:end_idx_8k]
            # The slice resampled_flushed_data_8kHz[A:B] will take min(B, len(array)) as end, so it's safe if end_idx_8k is too large.
        else:
            final_segment_float = np.array([], dtype=np.float32)

        self._reset_state() # Reset state after flushing
        
        if final_segment_float.size == 0:
            return None

        # Convert to µ-law
        pcm_int16_segment = (final_segment_float * 32767).astype(np.int16)
        if pcm_int16_segment.size == 0:
            return None
        pcm_int16_bytes = pcm_int16_segment.tobytes()
        ulaw_bytes = audioop.lin2ulaw(pcm_int16_bytes, 2)
        
        return base64.b64encode(ulaw_bytes).decode('utf-8')