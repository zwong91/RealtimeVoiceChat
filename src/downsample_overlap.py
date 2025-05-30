import base64
import numpy as np
from scipy.signal import resample_poly
import audioop  # For u-law conversion
from typing import Optional

class ResampleOverlapUlaw:
    """
    Manages chunk-wise audio downsampling (24kHz to 8kHz) with overlap handling.

    This class processes sequential audio chunks, downsamples them from 24kHz to 8kHz
    using `scipy.signal.resample_poly`, and manages overlap between chunks to
    mitigate boundary artifacts. The processed, downsampled audio segments are
    converted to u-law and returned as Base64 encoded strings. It maintains
    internal state to handle the overlap correctly across calls.
    """
    def __init__(self, input_fs: int = 24000, output_fs: int = 8000):
        """
        Initializes the ResampleOverlapUlaw processor.

        Args:
            input_fs: Input sampling frequency in Hz.
            output_fs: Output sampling frequency in Hz.
        """
        self.input_fs = input_fs
        self.output_fs = output_fs
        self.previous_chunk: Optional[np.ndarray] = None
        self.resampled_previous_chunk: Optional[np.ndarray] = None

        # Padding to help filter settle for the very first chunk.
        # Rule of thumb: ~30-100 samples for typical resampling scenarios.
        # This depends on the filter length implicit in resample_poly.
        # For 24kHz -> 8kHz, a modest padding like 96 input samples (4ms) is often sufficient.
        self.initial_padding_samples_in: int = 192
        # self.end_padding_samples_in: int = 96 # For potentially cleaner flush (see note below)


    def get_base64_chunk(self, chunk: bytes) -> str:
        """
        Processes an incoming audio chunk, downsamples it, and returns the relevant segment as Base64.

        Converts the raw PCM bytes (assumed 16-bit signed integer) chunk to a
        float32 numpy array, normalizes it, and downsamples.
        It uses the previous chunk's data to create an overlap, resamples the
        combined audio, and extracts the central portion corresponding primarily
        to the current chunk, using overlap to smooth transitions. The state is
        updated for the next call. The extracted audio segment is converted
        to u-law, then 16-bit PCM bytes and returned as a Base64 encoded string.

        Args:
            chunk: Raw audio data bytes (PCM 16-bit signed integer format expected).

        Returns:
            A Base64 encoded string representing the downsampled u-law audio segment
            corresponding to the input chunk, adjusted for overlap. Returns an
            empty string if the input chunk is empty.
        """
        if not chunk: # Handle empty bytes input
             return ""

        audio_int16 = np.frombuffer(chunk, dtype=np.int16)
        if audio_int16.size == 0:
             return "" # Return empty string for empty input chunk

        audio_float = audio_int16.astype(np.float32) / 32768.0

        # This will be the version of the current chunk resampled in isolation,
        # stored for the next iteration's `self.resampled_previous_chunk`
        # and for length calculations / flush logic if it's the last chunk.
        # For a potentially cleaner flush, this could also be padded at its end:
        # current_audio_float_padded_end = np.concatenate((audio_float, np.zeros(self.end_padding_samples_in, dtype=np.float32)))
        # downsampled_current_chunk_for_state_temp = resample_poly(current_audio_float_padded_end, self.output_fs, self.input_fs)
        # padding_out_end = self.end_padding_samples_in * self.output_fs // self.input_fs
        # downsampled_current_chunk_for_state = downsampled_current_chunk_for_state_temp[:-padding_out_end if padding_out_end > 0 else None]
        # -- Simpler version without end padding for now:
        downsampled_current_chunk_for_state = resample_poly(audio_float, self.output_fs, self.input_fs)


        if self.previous_chunk is None:
            # First chunk: Apply initial padding to audio_float before resampling
            # to get a cleaner version of downsampled_current_chunk.
            padded_audio_float = np.concatenate(
                (np.zeros(self.initial_padding_samples_in, dtype=np.float32), audio_float)
            )
            downsampled_padded_first_chunk = resample_poly(padded_audio_float, self.output_fs, self.input_fs)
            
            padding_samples_out = self.initial_padding_samples_in * self.output_fs // self.input_fs
            
            # This is the clean version of the first chunk, resampled
            clean_downsampled_first_chunk = downsampled_padded_first_chunk[padding_samples_out:]

            # Output the first 1/2 of this clean downsampled first chunk
            # (as per original strategy's division)
            overlap_len_out = len(clean_downsampled_first_chunk) // 2
            part = clean_downsampled_first_chunk[:overlap_len_out]
            
            # For state, use the consistently (unpadded here, or padded for flush) resampled current chunk
            self.resampled_previous_chunk = downsampled_current_chunk_for_state

        else:
            # Subsequent chunks: Combine previous float chunk with current float chunk
            combined = np.concatenate((self.previous_chunk, audio_float))
            # downsample the combined chunk
            up_combined_resampled = resample_poly(combined, self.output_fs, self.input_fs)

            assert self.resampled_previous_chunk is not None # Should be set from previous iteration
            
            # Length of the *individually resampled* previous chunk
            # (this is D(C_prev_isolated) and is used as reference for splitting up_combined_resampled)
            prev_resampled_isolated_len = len(self.resampled_previous_chunk)
            
            # Start index for extraction from up_combined_resampled:
            # Skip the first 1/2 of D(C_prev_isolated)'s length.
            # This part corresponds to D(C_prev_in_context_of_C_curr)[prev_resampled_isolated_len//2 : prev_resampled_isolated_len]
            h_prev = prev_resampled_isolated_len // 2

            # End index for extraction:
            # This determines how much of D(C_curr_in_context_of_C_prev) to take.
            # (len(up_combined_resampled) - prev_resampled_isolated_len) is approx. len(D(C_curr_isolated))
            # We take the first 1/2 of this part.
            current_chunk_contrib_len_in_up = len(up_combined_resampled) - prev_resampled_isolated_len
            h_cur = prev_resampled_isolated_len + (current_chunk_contrib_len_in_up // 2)
            
            part = up_combined_resampled[h_prev:h_cur]

            # For state, use the consistently (unpadded here) resampled current chunk
            self.resampled_previous_chunk = downsampled_current_chunk_for_state

        # Update state for the next iteration
        self.previous_chunk = audio_float
        # self.resampled_previous_chunk is already updated above based on context

        if part.size == 0: # Ensure we don't try to process an empty array
            return ""

        # Convert the extracted part back to PCM16 bytes, then u-law, then Base64
        pcm_part = (part * 32767).astype(np.int16).tobytes()
        ulaw_data = audioop.lin2ulaw(pcm_part, 2)  # 2 bytes per sample (16-bit)
        return base64.b64encode(ulaw_data).decode('utf-8')

    def flush_base64_chunk(self) -> Optional[str]:
        """
        Returns the final remaining segment of downsampled audio after all chunks are processed.

        This method returns the remaining part (typically last 2/3) of the
        final downsampled chunk (`self.resampled_previous_chunk`), converted
        to u-law and encoded as Base64. It then clears the internal state.
        This should be called once after all input chunks have been passed to `get_base64_chunk`.

        Returns:
            A Base64 encoded string containing the final u-law audio segment,
            or None if no chunks were processed or if flush has already been called.
        """
        if self.resampled_previous_chunk is not None and self.resampled_previous_chunk.size > 0:
            # Output the remaining 1/2 of the last processed chunk
            overlap_len_out = len(self.resampled_previous_chunk) // 2
            
            # Ensure we don't take a negative slice if overlap_len_out is 0 for very small chunks
            if overlap_len_out == 0 and len(self.resampled_previous_chunk) > 0: 
                # If chunk is too small for 1/2 overlap, flush what's left (which might be all of it if it's tiny)
                # This case should be rare with typical audio chunk sizes.
                # The first part (if any) would have been output in get_base64_chunk.
                # If it's the only chunk, [:0] was output, so flush all.
                # If it's a subsequent chunk, [:Ld//2] of it was output.
                # For simplicity and robustness with tiny chunks, if Ld//2 is 0,
                # we assume the logic in get_base64_chunk for h_prev:h_cur effectively handled it.
                # The most consistent is to stick to the rule:
                final_part_to_output = self.resampled_previous_chunk[overlap_len_out:]
            elif overlap_len_out > 0 :
                 final_part_to_output = self.resampled_previous_chunk[overlap_len_out:]
            else: # resampled_previous_chunk.size is 0, or overlap_len_out leads to empty
                final_part_to_output = np.array([], dtype=self.resampled_previous_chunk.dtype)


            if final_part_to_output.size == 0:
                # Clear state even if nothing to output from flush
                self.previous_chunk = None
                self.resampled_previous_chunk = None
                return None

            pcm_flush = (final_part_to_output * 32767).astype(np.int16).tobytes()
            
            # Clear state after flushing
            self.previous_chunk = None
            self.resampled_previous_chunk = None
            
            ulaw_data = audioop.lin2ulaw(pcm_flush, 2)
            return base64.b64encode(ulaw_data).decode('utf-8')
        
        # Clear state if it wasn't cleared already
        self.previous_chunk = None
        self.resampled_previous_chunk = None
        return None