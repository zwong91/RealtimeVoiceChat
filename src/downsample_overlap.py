import base64
import numpy as np
from scipy.signal import resample_poly, windows
import audioop
from typing import Optional

class ResampleOverlap:
    def __init__(self, input_fs: int = 24000, output_fs: int = 8000, overlap_ms: int = 20):
        self.input_fs = input_fs
        self.output_fs = output_fs
        self.overlap_samples_in = int(input_fs * overlap_ms / 1000)
        self.overlap_samples_out = int(output_fs * overlap_ms / 1000)

        self.previous_input_chunk = None
        self.previous_output_chunk = None

        self.window_length = 2 * self.overlap_samples_out
        # Ensure window_length is at least 1 if overlap_samples_out is 0
        if self.window_length == 0:
            self.overlap_add_window = np.array([1.0]) # Or handle the case where overlap_ms is 0
        else:
            self.overlap_add_window = windows.hann(self.window_length, sym=False)

        self.kaiser_beta = 14
        self.resample_filter_window = ('kaiser', self.kaiser_beta)

    def get_base64_chunk(self, chunk: bytes) -> str:
        if not chunk:
            return ""

        audio_int16 = np.frombuffer(chunk, dtype=np.int16)
        current_input_float = audio_int16.astype(np.float32) / 32768.0

        # This variable will hold the resampled output for the current input chunk *before* overlap-add blending
        current_resampled_segment = None

        if self.previous_input_chunk is None:
            # For the very first chunk, just resample it. No overlap to handle yet.
            current_resampled_segment = resample_poly(
                current_input_float,
                self.output_fs,
                self.input_fs,
                window=self.resample_filter_window
            )
            # The processed_output_chunk for the first iteration is simply the resampled_current
            processed_output_chunk = current_resampled_segment

        else:
            # Combine previous *input* chunk's tail with current *input* chunk's head
            combined_input = np.concatenate([self.previous_input_chunk[-self.overlap_samples_in:], current_input_float])

            # Resample the combined input segment
            current_resampled_segment = resample_poly(
                combined_input,
                self.output_fs,
                self.input_fs,
                window=self.resample_filter_window
            )

            # Now, apply overlap-add logic to the *resampled* outputs
            # Check if there's enough data to perform overlap-add
            if self.overlap_samples_out > 0 and \
               len(self.previous_output_chunk) >= self.overlap_samples_out and \
               len(current_resampled_segment) >= self.overlap_samples_out:

                # Segment of previous_output_chunk that overlaps with current
                prev_overlap_segment = self.previous_output_chunk[-self.overlap_samples_out:]

                # Segment of current_resampled_segment that corresponds to the overlap
                current_overlap_segment = current_resampled_segment[:self.overlap_samples_out]

                # Ensure window length matches overlap_samples_out
                # Recalculate if window_length was 0 initially and overlap_samples_out becomes > 0
                if self.window_length != 2 * self.overlap_samples_out:
                     self.window_length = 2 * self.overlap_samples_out
                     self.overlap_add_window = windows.hann(self.window_length, sym=False)

                # Ensure window segments match the overlap length
                window_first_half = self.overlap_add_window[:self.overlap_samples_out]
                window_second_half = self.overlap_add_window[self.overlap_samples_out:]

                # Apply windowing and add
                blended_overlap = (prev_overlap_segment * window_first_half) + \
                                  (current_overlap_segment * window_second_half)

                # Concatenate the non-overlapping part of the previous chunk, the blended overlap,
                # and the non-overlapping part of the current chunk.
                processed_output_chunk = np.concatenate([
                    self.previous_output_chunk[:-self.overlap_samples_out],
                    blended_overlap,
                    current_resampled_segment[self.overlap_samples_out:]
                ])
            else:
                # If overlap_samples_out is 0 or chunks are too small, just return the current resampled segment
                processed_output_chunk = current_resampled_segment


        self.previous_input_chunk = current_input_float
        # Store the entire resampled output of the *current* processing step for the next overlap
        # This should always be assigned after either the if or else block
        self.previous_output_chunk = current_resampled_segment

        clipped = np.clip(processed_output_chunk, -1.0, 1.0)
        int16_audio = (clipped * 32767.0).astype(np.int16).tobytes()
        ulaw_audio = audioop.lin2ulaw(int16_audio, 2)
        return base64.b64encode(ulaw_audio).decode("utf-8")

    def flush_base64_chunk(self) -> Optional[str]:
        if self.previous_output_chunk is not None and self.previous_output_chunk.size > 0:
            # Apply a fade-out to the last chunk
            # Ensure fade_out_length doesn't exceed the chunk size
            fade_out_length = min(self.overlap_samples_out * 2, len(self.previous_output_chunk))
            if fade_out_length > 0:
                # Use a smoother fade-out, e.g., second half of a Hann window, reversed
                fade_window = windows.hann(2 * fade_out_length, sym=False)[fade_out_length:]
                # Make sure fade_window length matches fade_out_length (slice may not be exact)
                fade_window = np.flip(fade_window) # Reverse to fade out from 1 to 0
                fade_window = fade_window[:fade_out_length] # Trim if necessary

                final_chunk = self.previous_output_chunk.copy()
                final_chunk[-fade_out_length:] *= fade_window
            else:
                final_chunk = self.previous_output_chunk

            clipped = np.clip(final_chunk, -1.0, 1.0)
            int16_audio = (clipped * 32767.0).astype(np.int16).tobytes()
            ulaw_audio = audioop.lin2ulaw(int16_audio, 2)
            self.previous_output_chunk = None
            self.previous_input_chunk = None
            return base64.b64encode(ulaw_audio).decode("utf-8")
        return None