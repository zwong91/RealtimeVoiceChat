import base64
import numpy as np
from scipy.signal import resample_poly, windows
import audioop
from typing import Optional

class ResampleOverlap:
    def __init__(self, input_fs: int = 24000, output_fs: int = 8000, overlap_ms: int = 20): # Increased overlap for better blending
        self.input_fs = input_fs
        self.output_fs = output_fs
        self.overlap_samples_in = int(input_fs * overlap_ms / 1000)
        self.overlap_samples_out = int(output_fs * overlap_ms / 1000)

        self.previous_input_chunk = None # Stores the original input chunk (float)
        self.previous_output_chunk = None # Stores the *resampled* chunk from the previous step

        # Use a Hann window for overlap-add, applied *after* resampling
        # The window length for overlap-add is usually twice the overlap
        self.window_length = 2 * self.overlap_samples_out
        self.overlap_add_window = windows.hann(self.window_length, sym=False)

        # Resampling window: Kaiser can still be good for anti-aliasing within resample_poly
        self.kaiser_beta = 14
        self.resample_filter_window = ('kaiser', self.kaiser_beta)

    def get_base64_chunk(self, chunk: bytes) -> str:
        if not chunk:
            return ""

        audio_int16 = np.frombuffer(chunk, dtype=np.int16)
        current_input_float = audio_int16.astype(np.float32) / 32768.0

        processed_output_chunk = np.array([], dtype=np.float32)

        if self.previous_input_chunk is None:
            # For the very first chunk, just resample it. No overlap to handle yet.
            resampled_current = resample_poly(
                current_input_float,
                self.output_fs,
                self.input_fs,
                window=self.resample_filter_window
            )
            processed_output_chunk = resampled_current

        else:
            # Combine previous *input* chunk's tail with current *input* chunk's head
            # for resampling to ensure no data loss at the boundary for the filter
            combined_input = np.concatenate([self.previous_input_chunk[-self.overlap_samples_in:], current_input_float])

            # Resample the combined input segment
            resampled_combined = resample_poly(
                combined_input,
                self.output_fs,
                self.input_fs,
                window=self.resample_filter_window
            )

            # Now, apply overlap-add logic to the *resampled* outputs
            # The resampled_combined now contains the resampled overlap portion at its beginning
            # We need to correctly segment resampled_combined.
            # The length of the part corresponding to the previous chunk's tail is self.overlap_samples_out.
            # The length of the part corresponding to the current chunk is len(resampled_combined) - self.overlap_samples_out.

            # The part of resampled_combined that overlaps with previous_output_chunk
            overlap_resampled_current_head = resampled_combined[:self.overlap_samples_out]

            # The new, non-overlapping part of the current chunk
            non_overlap_resampled_current_tail = resampled_combined[self.overlap_samples_out:]

            # Apply windowing for overlap-add
            # The window goes from 0 to 1 for the previous chunk's overlap region
            # and from 1 to 0 for the current chunk's overlap region.
            # Ensure window length matches overlap_samples_out
            if self.overlap_samples_out > 0:
                # The window should be applied to the overlapping part of the previous *output* chunk
                # and the overlapping part of the *current* resampled chunk.

                # Segment of previous_output_chunk that overlaps with current
                prev_overlap_segment = self.previous_output_chunk[-self.overlap_samples_out:]

                # Segment of current resampled_combined that corresponds to the overlap
                current_overlap_segment = resampled_combined[:self.overlap_samples_out]

                # Create two halves of the Hann window
                window_first_half = self.overlap_add_window[:self.overlap_samples_out]
                window_second_half = self.overlap_add_window[self.overlap_samples_out:]

                # Apply windowing and add
                # Note: This is a simplified overlap-add. A more robust method would involve
                # carefully selecting segments and adding. For now, let's try blending.
                blended_overlap = (prev_overlap_segment * window_first_half) + \
                                  (current_overlap_segment * window_second_half)

                # Concatenate the non-overlapping part of the previous chunk, the blended overlap,
                # and the non-overlapping part of the current chunk.
                processed_output_chunk = np.concatenate([
                    self.previous_output_chunk[:-self.overlap_samples_out],
                    blended_overlap,
                    resampled_combined[self.overlap_samples_out:]
                ])
            else:
                processed_output_chunk = resampled_combined


        self.previous_input_chunk = current_input_float
        # Store the entire resampled output of the *current* processing step for the next overlap
        self.previous_output_chunk = resampled_combined # This is the full resampled segment that *includes* the overlap part

        clipped = np.clip(processed_output_chunk, -1.0, 1.0)
        int16_audio = (clipped * 32767.0).astype(np.int16).tobytes()
        ulaw_audio = audioop.lin2ulaw(int16_audio, 2)
        return base64.b64encode(ulaw_audio).decode("utf-8")

    def flush_base64_chunk(self) -> Optional[str]:
        if self.previous_output_chunk is not None and self.previous_output_chunk.size > 0:
            # Apply a fade-out to the last chunk
            fade_out_length = min(self.overlap_samples_out * 2, len(self.previous_output_chunk))
            if fade_out_length > 0:
                fade_window = np.linspace(1.0, 0.0, fade_out_length)
                final_chunk = self.previous_output_chunk.copy()
                final_chunk[-fade_out_length:] *= fade_window
            else:
                final_chunk = self.previous_output_chunk

            clipped = np.clip(final_chunk, -1.0, 1.0)
            int16_audio = (clipped * 32767.0).astype(np.int16).tobytes()
            ulaw_audio = audioop.lin2ulaw(int16_audio, 2)
            self.previous_output_chunk = None # Reset for next stream
            self.previous_input_chunk = None # Reset for next stream
            return base64.b64encode(ulaw_audio).decode("utf-8")
        return None