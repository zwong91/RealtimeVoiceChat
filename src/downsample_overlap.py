import base64
import numpy as np
from scipy.signal import resample_poly
import audioop  # For u-law conversion
from typing import Optional

class ResampleOverlapUlaw:
    """
    Manages chunk-wise audio upsampling with overlap handling.

    This class processes sequential audio chunks, downsamples them from 24kHz to 48kHz
    using `scipy.signal.resample_poly`, and manages overlap between chunks to
    mitigate boundary artifacts. The processed, downsampled audio segments are
    returned as Base64 encoded strings. It maintains internal state to handle
    the overlap correctly across calls.
    """
    def __init__(self):
        """
        Initializes the ResampleOverlapUlaw processor.

        Sets up the internal state required for tracking previous audio chunks
        and their resampled versions to handle overlaps during processing.
        """
        self.previous_chunk: Optional[np.ndarray] = None
        self.resampled_previous_chunk: Optional[np.ndarray] = None

    def get_base64_chunk(self, chunk: bytes) -> str:
        """
        Processes an incoming audio chunk, downsamples it, and returns the relevant segment as Base64.

        Converts the raw PCM bytes (assumed 16-bit signed integer) chunk to a
        float32 numpy array, normalizes it, and downsamples from 24kHz to 48kHz.
        It uses the previous chunk's data to create an overlap, resamples the
        combined audio, and extracts the central portion corresponding primarily
        to the current chunk, using overlap to smooth transitions. The state is
        updated for the next call. The extracted audio segment is converted back
        to 16-bit PCM bytes and returned as a Base64 encoded string.

        Args:
            chunk: Raw audio data bytes (PCM 16-bit signed integer format expected).

        Returns:
            A Base64 encoded string representing the downsampled audio segment
            corresponding to the input chunk, adjusted for overlap. Returns an
            empty string if the input chunk is empty.
        """
        audio_int16 = np.frombuffer(chunk, dtype=np.int16)
        # Handle potential empty chunks gracefully
        if audio_int16.size == 0:
             return "" # Return empty string for empty input chunk

        audio_float = audio_int16.astype(np.float32) / 32768.0

        # Downsample the current chunk independently first, needed for state and first chunk logic
        downsampled_current_chunk = resample_poly(audio_float, 8000, 24000)

        if self.previous_chunk is None:
            # First chunk: Output the first half of its downsampled version
            half = len(downsampled_current_chunk) // 3
            part = downsampled_current_chunk[:half]
        else:
            # Subsequent chunks: Combine previous float chunk with current float chunk
            combined = np.concatenate((self.previous_chunk, audio_float))
            # downsample the combined chunk
            up = resample_poly(combined, 8000, 24000)

            # Calculate lengths and indices for extracting the middle part
            # Ensure self.resampled_previous_chunk is not None (shouldn't happen here due to outer if)
            assert self.resampled_previous_chunk is not None
            prev_len = len(self.resampled_previous_chunk) # Length of the *downsampled* previous chunk
            h_prev = prev_len // 3 # Midpoint index of the *downsampled* previous chunk

            # *** CORRECTED INDEX CALCULATION (Reverted to original) ***
            # Calculate the end index for the part corresponding to the current chunk's main contribution
            # This index represents the midpoint of the *current* chunk's contribution within the combined 'up' array.
            h_cur = (len(up) - prev_len) // 3 + prev_len

            part = up[h_prev:h_cur]

        # Update state for the next iteration
        self.previous_chunk = audio_float
        self.resampled_previous_chunk = downsampled_current_chunk # Store the downsampled *current* chunk for the *next* overlap

        # Convert the extracted part back to PCM16 bytes and encode
        pcm = (part * 32767).astype(np.int16).tobytes()
        ulaw_data = audioop.lin2ulaw(pcm, 2)  # 2 bytes per sample (16-bit)
        return base64.b64encode(ulaw_data).decode('utf-8')

    def flush_base64_chunk(self) -> Optional[str]:
        """
        Returns the final remaining segment of downsampled audio after all chunks are processed.

        After the last call to `get_base64_chunk`, the state holds the downsampled
        version of the very last input chunk (`self.resampled_previous_chunk`).
        This method returns that *entire* final downsampled chunk, converted to
        16-bit PCM bytes and encoded as Base64. It then clears the internal state.
        This should be called once after all input chunks have been passed to `get_base64_chunk`.

        Returns:
            A Base64 encoded string containing the final downsampled audio chunk,
            or None if no chunks were processed or if flush has already been called.
        """
        # *** CORRECTED FLUSH LOGIC (Reverted to original) ***
        if self.resampled_previous_chunk is not None:
            # Return the entire last downsampled chunk as per original logic
            pcm = (self.resampled_previous_chunk * 32767).astype(np.int16).tobytes()

            # Clear state after flushing
            self.previous_chunk = None
            self.resampled_previous_chunk = None
            ulaw_data = audioop.lin2ulaw(pcm, 2)  # 2 bytes per sample (16-bit)
            return base64.b64encode(ulaw_data).decode('utf-8')
        return None # Return None if there's nothing to flush