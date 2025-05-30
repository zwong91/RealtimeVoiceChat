import base64
import numpy as np
from scipy.signal import resample_poly
import audioop
from typing import Optional

class ResampleOverlapUlaw:
    def __init__(self, input_fs: int = 24000, output_fs: int = 8000):
        self.input_fs = input_fs
        self.output_fs = output_fs
        self.previous_raw_chunk: Optional[np.ndarray] = None # Store raw previous chunk
        
        # Overlap in terms of INPUT samples. This is easier to reason about for filter length.
        # Let's try a significant overlap, e.g., 20ms.
        # 20ms at 24kHz = 0.020 * 24000 = 480 samples.
        # This overlap will be applied to BOTH sides of a chunk when it's in the "middle"
        # of a previous and next chunk during combined resampling.
        self.overlap_input_samples: int = 240 # Try 10ms overlap (240 samples @24kHz)
                                            # This means we take previous_chunk[-overlap:] + current_chunk + next_chunk[:overlap]
                                            # For streaming, we'll use previous_chunk[-overlap:] + current_chunk

        self.buffer = np.array([], dtype=np.float32) # To store resampled audio
        self.first_chunk_processed = False

        # Initial padding for the very first segment
        self.initial_padding_in = 192 # Keep this

    def _resample(self, audio_float: np.ndarray) -> np.ndarray:
        # Wrapper for resample_poly, potentially with custom window if needed later
        return resample_poly(audio_float, self.output_fs, self.input_fs)

    def get_base64_chunk(self, chunk: bytes) -> str:
        if not chunk:
            return ""
        current_raw_chunk = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        if current_raw_chunk.size == 0:
            return ""

        output_part = np.array([], dtype=np.float32)

        if not self.first_chunk_processed:
            # Handle the very first chunk with initial padding
            padded_input = np.concatenate((np.zeros(self.initial_padding_in, dtype=np.float32), current_raw_chunk))
            resampled_padded = self._resample(padded_input)
            
            padding_out = self.initial_padding_in * self.output_fs // self.input_fs
            resampled_first_clean = resampled_padded[padding_out:]
            
            # For the first chunk, we decide how much to output now vs buffer
            # Output roughly half, buffer the rest for next overlap
            # Or, more simply, output what corresponds to current_raw_chunk MINUS the overlap we'll need for the NEXT chunk
            
            # Estimate length of current_raw_chunk if resampled in isolation
            current_resampled_len_approx = current_raw_chunk.size * self.output_fs // self.input_fs
            overlap_out_samples_approx = self.overlap_input_samples * self.output_fs // self.input_fs
            
            # We need to keep 'overlap_out_samples_approx' from the end of resampled_first_clean
            # if resampled_first_clean is long enough
            if len(resampled_first_clean) > overlap_out_samples_approx:
                # output_now_len = len(resampled_first_clean) - overlap_out_samples_approx
                # A simpler "overlap-add" like approach might be better.
                # Output all of resampled_first_clean for now, and handle overlap on the next call.
                # This is simpler than trying to precisely cut the first chunk.
                # The "previous_raw_chunk" will be the key.
                self.buffer = np.concatenate((self.buffer, resampled_first_clean))
                
            else: # First chunk is very short, buffer all of it
                 self.buffer = np.concatenate((self.buffer, resampled_first_clean))

            self.first_chunk_processed = True
        
        else: # Subsequent chunks
            assert self.previous_raw_chunk is not None
            # Take the end of the previous chunk and the current chunk
            # The overlap region is from previous_raw_chunk
            overlap_region = self.previous_raw_chunk[-self.overlap_input_samples:] if len(self.previous_raw_chunk) >= self.overlap_input_samples else self.previous_raw_chunk
            
            # Segment to resample: overlap_from_prev + current_chunk
            segment_to_resample = np.concatenate((overlap_region, current_raw_chunk))
            resampled_segment = self._resample(segment_to_resample)
            
            # How much of this resampled_segment corresponds to the 'overlap_region' part?
            resampled_overlap_len = overlap_region.size * self.output_fs // self.input_fs
            
            # The new audio we want is from resampled_segment, starting AFTER the resampled_overlap_len
            # This is the main contribution of current_raw_chunk
            new_audio = resampled_segment[resampled_overlap_len:]
            self.buffer = np.concatenate((self.buffer, new_audio))

        # Now, decide how much to take from self.buffer to output
        # We always want to keep roughly `overlap_input_samples` (converted to output rate)
        # in the buffer for the *next* processing step, unless we are flushing.
        
        # This logic becomes more like a standard overlap-add/save
        # Let's simplify: for streaming, we output based on the original chunk size's expected output
        # This avoids complex buffer management for get_chunk, and flush handles the rest.
        
        # Reverting to a slightly modified version of your 50% logic for output, but with state based on raw chunks
        # The problem might be that `self.resampled_previous_chunk` was resampled in isolation.
        
        # Let's go back to your structure and try to pinpoint the issue.
        # The most common cause of clicks is if the segments `part` don't perfectly align end-to-start.
        # This can happen if `len(D(A+B))` is not exactly `len(D(A)) + len(D(B))` due to filter tails.

        # Re-inserting your original code structure here for direct modification/commentary
        # (Your provided code is already in the prompt)

        # One key difference in robust overlap-add/save is that the analysis/synthesis windows
        # sum to a constant in the overlap regions. `resample_poly` doesn't inherently do this for you
        # when you manually chunk.

        # Try to ensure `part` starts exactly where the previous `part` ended, conceptually.
        # The issue is that `self.resampled_previous_chunk` (D(C_prev_isolated))
        # and the D(C_prev) portion within `up_combined_resampled` (D(C_prev + C_curr))
        # might not be identical due to context.
        # `D(C_prev_isolated)` has filter transients at its start and end.
        # `D(C_prev_in_context)` (the C_prev part of `up_combined_resampled`) has transients
        # at the start of `C_prev` (within `combined`) and at the end of `C_curr` (within `combined`).

        # If `initial_padding_samples_in` is large enough, `clean_downsampled_first_chunk` should be good.
        # Let's assume `part = clean_downsampled_first_chunk[:overlap_len_out]` is good.
        # For the next chunk:
        # `part = up_combined_resampled[h_prev:h_cur]`
        # `h_prev = prev_resampled_isolated_len // 2`
        # `h_cur = prev_resampled_isolated_len + (current_chunk_contrib_len_in_up // 2)`
        # The segment from `up_combined_resampled` that is `up_combined_resampled[:prev_resampled_isolated_len]`
        # is the D(C_prev) part *when resampled in context with C_curr*.
        # Its second half is `up_combined_resampled[prev_resampled_isolated_len // 2 : prev_resampled_isolated_len]`.
        # The segment from `up_combined_resampled` that is `up_combined_resampled[prev_resampled_isolated_len:]`
        # is the D(C_curr) part *when resampled in context with C_prev*.
        # Its first half is `up_combined_resampled[prev_resampled_isolated_len : prev_resampled_isolated_len + current_chunk_contrib_len_in_up // 2]`.

        # So, `part` is `D(C_prev_in_context)[後半] + D(C_curr_in_context)[前半]`.
        # The previous `part` was `D(C_prev_isolated_padded_at_start_if_first_chunk_else_C_prev_minus_1_in_context)[前半]`.

        # The click might be between:
        # Output1: `D(C1_padded)[:len1//2]`
        # Output2: `D(C1+C2)[len1//2 : len1 + len2_in_C1C2//2]`
        # The problem is `D(C1_padded)[:len1//2]` and `D(C1+C2)[:len1//2]` might not be identical due to C2's influence.

        # A standard overlap-save method:
        # 1. Buffer input samples: `buffer = previous_chunk_tail + current_chunk`
        # 2. Resample `buffer` -> `resampled_buffer`
        # 3. Discard initial part of `resampled_buffer` (corresponding to filter transient + previous_chunk_tail's resampled part that was already accounted for or is part of the overlap to discard).
        # 4. Output the valid part of `resampled_buffer`.
        # 5. Update `previous_chunk_tail`.

        # Let's try to make your current code more like overlap-save.
        # `self.previous_chunk` is `audio_float` (raw C_prev)
        # `self.resampled_previous_chunk` is `downsampled_current_chunk_for_state` (D(C_prev_isolated)) - this might be the problem.

        # What if `self.resampled_previous_chunk` stored the *part that was just outputted*? No, that's not right.

        # Let's simplify `get_base64_chunk` state: only `self.previous_raw_overlap_samples_in`
        # These are raw input samples from the tail of the previous chunk that we need to prepend.
        self.raw_overlap_to_prepend: np.ndarray = np.array([], dtype=np.float32)
        self.initial_padding_in_actual_for_first_call = self.initial_padding_samples_in # Use a mutable copy

        # Resetting the class for a cleaner OLA-like approach trial
        if "_ola_initialized" not in self.__dict__: # One-time init for OLA state
            self.ola_buffer_in: np.ndarray = np.zeros(self.initial_padding_samples_in, dtype=np.float32) # Start with padding
            self.ola_samples_to_discard_at_output_start: int = self.initial_padding_samples_in * self.output_fs // self.input_fs
            self._ola_initialized = True


        # OLA-like approach attempt:
        # Concatenate the necessary previous context with the current chunk
        processing_block_in = np.concatenate((self.ola_buffer_in, audio_float))
        
        resampled_block_out = resample_poly(processing_block_in, self.output_fs, self.input_fs)

        # Discard samples from the beginning of resampled_block_out
        # These correspond to the initial padding (first call) or the overlap from previous (subsequent calls)
        # that we don't want to output again or that are part of the filter settling.
        valid_part_offset = self.ola_samples_to_discard_at_output_start
        
        # On the very first call, initial_padding_in results in padding_out to discard.
        # On subsequent calls, ola_samples_to_discard_at_output_start should correspond to
        # the resampled version of the overlap region we prepended.

        part_to_output = resampled_block_out[valid_part_offset:]
        
        # Update `ola_buffer_in` for the next iteration:
        # It should be the tail of the *current* `audio_float` that will serve as overlap.
        # The length of this tail should be such that its resampled version is what we'd discard.
        # A common overlap for OLA is 50% of the FFT window, but here it's for FIR filter.
        # Let's use a fixed number of input samples for overlap, e.g., `self.initial_padding_samples_in`
        # This many samples from the end of `audio_float` will be `ola_buffer_in` for next time.
        num_overlap_samples_in = self.initial_padding_samples_in # Reuse this, it's a decent filter settling length

        if len(audio_float) >= num_overlap_samples_in:
            self.ola_buffer_in = audio_float[-num_overlap_samples_in:]
        else: # current chunk is shorter than desired overlap
            self.ola_buffer_in = audio_float 
        
        # For the next call, the amount to discard from output is the resampled length of this new `ola_buffer_in`
        self.ola_samples_to_discard_at_output_start = len(self.ola_buffer_in) * self.output_fs // self.input_fs
        
        # --- End of OLA-like approach ---
        # This OLA approach is simpler and often more robust for clicks.
        # Let's use `part_to_output` from this OLA logic.

        # Original state updates (remove if fully committing to OLA above)
        # self.previous_chunk = audio_float
        # self.resampled_previous_chunk = downsampled_current_chunk_for_state # This is D(current_isolated)

        # If using OLA, the `part` is `part_to_output`
        part = part_to_output # From the OLA section

        if part.size == 0:
            # This can happen if audio_float is very short and after discarding overlap, nothing is left.
            # This might mean the input chunks are too small for this overlap amount.
            # Or if it's the end and flush hasn't been called.
            # For now, return empty, but this needs robust handling with flush.
            return ""

        pcm = (part * 32767).astype(np.int16).tobytes()
        ulaw_data = audioop.lin2ulaw(pcm, 2)
        return base64.b64encode(ulaw_data).decode('utf-8')

    def flush_base64_chunk(self) -> Optional[str]:
        """
        In an OLA (Overlap-Add or Overlap-Save) context, flush typically means processing
        any remaining samples in a final buffer, possibly with zero-padding at the end
        to allow the filter to fully run down.

        The OLA approach above is more like Overlap-Save where `ola_buffer_in` holds
        the necessary history. When `get_base64_chunk` is no longer called, there's no
        "new" audio to append. The `ola_buffer_in` itself doesn't typically get "flushed"
        as a standalone piece in OLA-Save because its corresponding output was part of the
        *previous* `get_base64_chunk` call.

        If the last `part_to_output` from `get_base64_chunk` was complete for the
        final actual audio chunk, then flush might not need to do anything *extra*
        beyond clearing state.

        However, `resample_poly` has a filter tail. If the *very last* piece of audio
        was `audio_float` in the last `get_base64_chunk` call, `part_to_output` would
        be `resampled_block_out[valid_part_offset:]`. This `resampled_block_out` was
        from `concatenate((self.ola_buffer_in, audio_float))`. The filter tail for
        `audio_float` is included.

        So, the OLA flush might just be a state clear.
        Let's test the OLA version first. The original flush logic is tied to `self.resampled_previous_chunk`.
        """
        # OLA FLUSH:
        # The OLA method above ensures that for each input chunk, the corresponding output is generated.
        # The `self.ola_buffer_in` is state for the *next* call. If no next call, it's not used.
        # The filter's tail for the *last* actual audio data processed in the final get_base64_chunk
        # call should have been included in that call's output.
        # So, an OLA flush might just be clearing internal OLA state.
        if hasattr(self, '_ola_initialized'): # Check if OLA was used
            self.ola_buffer_in = np.array([], dtype=np.float32)
            self.ola_samples_to_discard_at_output_start = 0
            del self._ola_initialized # Reset for potential re-use of instance
            # For OLA, there might be nothing "extra" to flush if get_chunk handled last segment fully.
            # However, the original design expected `resampled_previous_chunk` to be flushed.
            # If we stick to OLA, the concept of flushing `resampled_previous_chunk` (D(C_last_isolated)) doesn't quite fit.

        # Reverting to your original flush logic for now, as the OLA part above is an alternative within get_base64_chunk
        # To use the OLA approach fully, flush would also change.
        # For now, let's assume the OLA section was a test and we fall back to your original structure
        # for the rest of get_base64_chunk and flush.
        # So, comment out the OLA specific flush part above.

        # ORIGINAL FLUSH LOGIC (as in your provided code)
        if self.resampled_previous_chunk is not None and self.resampled_previous_chunk.size > 0:
            overlap_len_out = len(self.resampled_previous_chunk) // 2 # Using 50%
            final_part_to_output = np.array([], dtype=self.resampled_previous_chunk.dtype)

            if overlap_len_out == 0 and len(self.resampled_previous_chunk) > 0 :
                final_part_to_output = self.resampled_previous_chunk[overlap_len_out:]
            elif overlap_len_out > 0 and overlap_len_out < len(self.resampled_previous_chunk):
                 final_part_to_output = self.resampled_previous_chunk[overlap_len_out:]
            # Ensure this condition is safe: if overlap_len_out >= len, it means slice is empty or invalid
            elif overlap_len_out >= len(self.resampled_previous_chunk) and len(self.resampled_previous_chunk) > 0:
                pass # final_part_to_output remains empty

            if final_part_to_output.size == 0:
                self.previous_chunk = None
                self.resampled_previous_chunk = None
                return None

            pcm_flush = (final_part_to_output * 32767).astype(np.int16).tobytes()
            self.previous_chunk = None
            self.resampled_previous_chunk = None
            ulaw_data = audioop.lin2ulaw(pcm_flush, 2)
            return base64.b64encode(ulaw_data).decode('utf-8')

        self.previous_chunk = None
        self.resampled_previous_chunk = None
        return None