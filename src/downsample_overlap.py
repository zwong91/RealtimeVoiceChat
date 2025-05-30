import base64
import numpy as np
from scipy.signal import resample_poly
from typing import Optional
import audioop # For µ-law conversion

class DownsampleMuLawOverlap:
    """
    Manages chunk-wise audio downsampling to 8kHz µ-law with overlap handling.

    This class processes sequential audio chunks (assumed 24kHz PCM16),
    downsamples them to 8kHz using `scipy.signal.resample_poly`,
    manages overlap between chunks to mitigate boundary artifacts, converts
    the audio to µ-law format, and returns segments as Base64 encoded strings.
    It maintains internal state to handle the overlap correctly across calls.
    """
    SOURCE_RATE = 24000
    TARGET_RATE = 8000

    def __init__(self):
        """
        Initializes the DownsampleMuLawOverlap processor.

        Sets up internal state for tracking previous audio chunks and their
        resampled versions to handle overlaps.
        """
        self.previous_chunk_float: Optional[np.ndarray] = None # Stores original 24kHz float audio
        self.resampled_previous_chunk_float: Optional[np.ndarray] = None # Stores 8kHz resampled float audio

    def get_base64_chunk(self, chunk: bytes) -> str:
        """
        Processes an audio chunk, downsamples it to 8kHz µ-law, and returns the segment as Base64.

        Converts raw PCM 16-bit signed integer bytes to a float32 numpy array,
        normalizes it, and downsamples from 24kHz to 8kHz.
        It uses the previous chunk's data for overlap, resamples the combined
        audio, and extracts the central portion corresponding primarily to the
        current chunk. The state is updated. The extracted audio segment is
        converted to µ-law bytes and returned as a Base64 encoded string.

        Args:
            chunk: Raw audio data bytes (PCM 16-bit signed integer, 24kHz expected).

        Returns:
            A Base64 encoded string representing the downsampled 8kHz µ-law audio
            segment, or an empty string if the input chunk is empty.
        """
        if not chunk:
            return ""

        audio_int16 = np.frombuffer(chunk, dtype=np.int16)
        if audio_int16.size == 0:
             return ""

        current_audio_float = audio_int16.astype(np.float32) / 32768.0

        # Downsample the current chunk independently first for state and first chunk logic
        resampled_current_chunk = resample_poly(current_audio_float, self.TARGET_RATE, self.SOURCE_RATE)

        output_segment_float: np.ndarray

        if self.previous_chunk_float is None:
            # First chunk: Output the first half of its resampled version
            # This prepares the state for the next chunk to use its second half for overlap.
            half_len = len(resampled_current_chunk) // 2
            output_segment_float = resampled_current_chunk[:half_len]
        else:
            # Subsequent chunks: Combine previous float chunk (24kHz) with current float chunk (24kHz)
            combined_float = np.concatenate((self.previous_chunk_float, current_audio_float))
            # Downsample the combined chunk to 8kHz
            resampled_combined = resample_poly(combined_float, self.TARGET_RATE, self.SOURCE_RATE)

            # Calculate lengths and indices for extracting the middle part
            # self.resampled_previous_chunk_float is the 8kHz resampled version of the *previous* input chunk
            assert self.resampled_previous_chunk_float is not None
            len_resampled_prev = len(self.resampled_previous_chunk_float)
            
            # Start index: Midpoint of the resampled previous chunk's contribution in 'resampled_combined'
            start_index = len_resampled_prev // 2
            
            # End index: Midpoint of the resampled current chunk's contribution in 'resampled_combined'
            # This ensures we take roughly one chunk's worth of new audio, centered around the join.
            # Length of current chunk's contribution to resampled_combined is (len(resampled_combined) - len_resampled_prev)
            end_index = len_resampled_prev + ( (len(resampled_combined) - len_resampled_prev) // 2 )
            
            output_segment_float = resampled_combined[start_index:end_index]

        # Update state for the next iteration
        self.previous_chunk_float = current_audio_float # Store original 24kHz float
        self.resampled_previous_chunk_float = resampled_current_chunk # Store 8kHz resampled float

        # Convert the extracted float segment (8kHz) to µ-law bytes
        # 1. Convert float32 (normalized) to int16
        pcm_int16_segment = (output_segment_float * 32767).astype(np.int16)
        # 2. Convert int16 array to bytes
        pcm_int16_bytes = pcm_int16_segment.tobytes()
        # 3. Convert linear PCM16 bytes to µ-law bytes (sample width is 2 bytes for int16)
        ulaw_bytes = audioop.lin2ulaw(pcm_int16_bytes, 2)
        
        return base64.b64encode(ulaw_bytes).decode('utf-8')

    def flush_base64_chunk(self) -> Optional[str]:
        """
        Returns the final remaining segment of 8kHz µ-law audio.

        After the last call to `get_base64_chunk`, this method processes the
        remaining portion (typically the second half) of the last resampled audio
        chunk. It converts this segment to µ-law and returns it as a Base64 string.
        Clears internal state.

        Returns:
            A Base64 encoded string of the final 8kHz µ-law audio segment,
            or None if no chunks were processed.
        """
        if self.resampled_previous_chunk_float is None:
            return None

        # The `get_base64_chunk` method outputs the "first half" of the
        # resampled_previous_chunk_float's contribution.
        # So, we need to output the "second half" here.
        len_last_resampled_chunk = len(self.resampled_previous_chunk_float)
        start_index_for_flush = len_last_resampled_chunk // 2
        
        final_segment_float = self.resampled_previous_chunk_float[start_index_for_flush:]

        if final_segment_float.size == 0:
            # Clear state even if the remaining part is empty
            self.previous_chunk_float = None
            self.resampled_previous_chunk_float = None
            return None # Or an empty Base64 string for an empty µ-law chunk, e.g., base64.b64encode(b'').decode()

        # Convert to µ-law
        pcm_int16_segment = (final_segment_float * 32767).astype(np.int16)
        pcm_int16_bytes = pcm_int16_segment.tobytes()
        ulaw_bytes = audioop.lin2ulaw(pcm_int16_bytes, 2)

        # Clear state after flushing
        self.previous_chunk_float = None
        self.resampled_previous_chunk_float = None
        
        return base64.b64encode(ulaw_bytes).decode('utf-8')