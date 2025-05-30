import base64
import numpy as np
import audioop
from scipy.signal import resample_poly, butter, filtfilt
from typing import Optional

class UpsampleOverlap:
    """
    Manages chunk-wise audio downsampling with overlap handling and anti-aliasing filtering.

    This class processes sequential audio chunks, applies anti-aliasing filtering,
    downsamples them from 24kHz to 8kHz using `scipy.signal.resample_poly`, and 
    manages overlap between chunks to mitigate boundary artifacts. The processed, 
    downsampled audio segments are converted to μ-law format and returned as Base64 
    encoded strings. It maintains internal state to handle the overlap correctly across calls.
    """
    def __init__(self):
        """
        Initializes the UpsampleOverlap processor.

        Sets up the internal state required for tracking previous audio chunks
        and their resampled versions to handle overlaps during processing.
        Also initializes the anti-aliasing filter coefficients.
        """
        self.previous_chunk: Optional[np.ndarray] = None
        self.resampled_previous_chunk: Optional[np.ndarray] = None
        
        # Anti-aliasing filter setup for 24kHz -> 8kHz downsampling
        self.nyquist = 24000 / 2  # 12kHz
        self.cutoff = 3800  # Slightly below 4kHz (Nyquist for 8kHz)
        self.filter_order = 5
        self.b, self.a = butter(self.filter_order, self.cutoff / self.nyquist, btype='low')

    def get_base64_chunk(self, chunk: bytes) -> str:
        """
        Processes an incoming audio chunk with anti-aliasing, downsamples it to 8kHz, 
        converts to μ-law, and returns as Base64.

        Converts the raw PCM bytes (assumed 16-bit signed integer) chunk to a
        float32 numpy array, normalizes it, applies anti-aliasing filtering,
        and downsamples from 24kHz to 8kHz. It uses the previous chunk's data 
        to create an overlap, resamples the combined audio, and extracts the 
        central portion corresponding primarily to the current chunk, using 
        overlap to smooth transitions. The state is updated for the next call. 
        The extracted audio segment is converted to μ-law format and returned 
        as a Base64 encoded string.

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
             return "" # Return empty string for empty input chunk

        audio_float = audio_int16.astype(np.float32) / 32768.0

        # Apply anti-aliasing filter before downsampling
        if len(audio_float) > 2 * self.filter_order:  # Ensure minimum length for filtering
            audio_filtered = filtfilt(self.b, self.a, audio_float)
        else:
            # For very short chunks, skip filtering to avoid artifacts
            audio_filtered = audio_float

        # Downsample the current chunk independently first, needed for state and first chunk logic
        downsampled_current_chunk = resample_poly(audio_filtered, 8000, 24000)

        if self.previous_chunk is None:
            # First chunk: Output the first half of its downsampled version
            half = len(downsampled_current_chunk) // 2
            part = downsampled_current_chunk[:half]
        else:
            # Subsequent chunks: Combine previous float chunk with current float chunk
            combined = np.concatenate((self.previous_chunk, audio_filtered))
            
            # Apply anti-aliasing filter to the combined chunk if long enough
            if len(combined) > 2 * self.filter_order:
                combined_filtered = filtfilt(self.b, self.a, combined)
            else:
                combined_filtered = combined
            
            # Downsample the combined chunk
            down = resample_poly(combined_filtered, 8000, 24000)

            # Calculate lengths and indices for extracting the middle part
            # Ensure self.resampled_previous_chunk is not None (shouldn't happen here due to outer if)
            assert self.resampled_previous_chunk is not None
            prev_len = len(self.resampled_previous_chunk) # Length of the *downsampled* previous chunk
            h_prev = prev_len // 2 # Midpoint index of the *downsampled* previous chunk

            # Calculate the end index for the part corresponding to the current chunk's main contribution
            # This index represents the midpoint of the *current* chunk's contribution within the combined 'down' array.
            h_cur = (len(down) - prev_len) // 2 + prev_len

            part = down[h_prev:h_cur]

        # Update state for the next iteration (store filtered version for consistency)
        self.previous_chunk = audio_filtered
        self.resampled_previous_chunk = downsampled_current_chunk # Store the downsampled *current* chunk for the *next* overlap

        # Convert the extracted part to PCM16 bytes, then directly convert to μ-law
        pcm16_bytes = (part * 32767).astype(np.int16).tobytes()
        ulaw_bytes = audioop.lin2ulaw(pcm16_bytes, 2)  # Direct PCM16 to μ-law conversion
        return base64.b64encode(ulaw_bytes).decode('utf-8')

    def flush_base64_chunk(self) -> Optional[str]:
        """
        Returns the final remaining segment of downsampled μ-law audio after all chunks are processed.

        After the last call to `get_base64_chunk`, the state holds the downsampled
        version of the very last input chunk (`self.resampled_previous_chunk`).
        This method returns that *entire* final downsampled chunk, converted to
        μ-law format and encoded as Base64. It then clears the internal state.
        This should be called once after all input chunks have been passed to `get_base64_chunk`.

        Returns:
            A Base64 encoded string containing the final downsampled μ-law audio chunk,
            or None if no chunks were processed or if flush has already been called.
        """
        if self.resampled_previous_chunk is not None:
            # Return the entire last downsampled chunk converted to μ-law
            pcm16_bytes = (self.resampled_previous_chunk * 32767).astype(np.int16).tobytes()
            ulaw_bytes = audioop.lin2ulaw(pcm16_bytes, 2)  # Direct PCM16 to μ-law conversion

            # Clear state after flushing
            self.previous_chunk = None
            self.resampled_previous_chunk = None
            return base64.b64encode(ulaw_bytes).decode('utf-8')
        return None # Return None if there's nothing to flush