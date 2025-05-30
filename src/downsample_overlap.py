import base64
import numpy as np
from scipy.signal import resample_poly, get_window # get_window for fade shapes
import audioop
from typing import Optional

class ResampleOverlapUlaw:
    def __init__(self, input_fs: int = 24000, output_fs: int = 8000,
                 fade_duration_ms: float = 5.0): # Fade duration in milliseconds
        self.input_fs = input_fs
        self.output_fs = output_fs
        self.previous_chunk: Optional[np.ndarray] = None
        self.resampled_previous_chunk: Optional[np.ndarray] = None
        self.initial_padding_samples_in: int = 192

        # Fade parameters
        self.fade_duration_ms = fade_duration_ms
        # Calculate fade length in samples AT THE OUTPUT sampling rate
        self.fade_length_out_samples: int = int(self.output_fs * (self.fade_duration_ms / 1000.0))
        
        # Ensure fade length is not too long, e.g., at least 2 samples for a ramp
        if self.fade_length_out_samples < 2:
            self.fade_length_out_samples = 0 # Disable fade if too short
        
        if self.fade_length_out_samples > 0:
            # Create fade windows (linear ramp for simplicity here)
            # self.fade_in_window = np.linspace(0.0, 1.0, self.fade_length_out_samples, dtype=np.float32)
            # self.fade_out_window = np.linspace(1.0, 0.0, self.fade_length_out_samples, dtype=np.float32)

            # Using a Hanning window for smoother fades
            hanning_half_window = get_window('hanning', self.fade_length_out_samples * 2, fftbins=False)
            self.fade_in_window = hanning_half_window[:self.fade_length_out_samples].astype(np.float32)
            self.fade_out_window = hanning_half_window[self.fade_length_out_samples:].astype(np.float32)[::-1] # Reverse the second half
        else:
            self.fade_in_window = np.array([], dtype=np.float32)
            self.fade_out_window = np.array([], dtype=np.float32)


    def _apply_fade(self, audio_segment: np.ndarray) -> np.ndarray:
        """Applies fade-in to the beginning and fade-out to the end of the segment."""
        if self.fade_length_out_samples == 0 or audio_segment.size == 0:
            return audio_segment

        segment_len = audio_segment.size
        
        # Ensure fade length is not longer than half the segment length
        # to prevent fades from overlapping completely or exceeding segment bounds.
        actual_fade_len = min(self.fade_length_out_samples, segment_len // 2)

        if actual_fade_len < 2 : # Not enough samples for a meaningful fade
            return audio_segment

        # Adjust windows if actual_fade_len is smaller than self.fade_length_out_samples
        # This requires re-calculating or slicing the pre-calculated windows.
        # For simplicity, if actual_fade_len is different, we re-calculate linear ramps.
        # Or, use the pre-calculated windows and apply them carefully.
        
        current_fade_in_window = self.fade_in_window
        current_fade_out_window = self.fade_out_window

        if actual_fade_len != self.fade_length_out_samples:
            # If segment is too short, recalculate shorter linear fades for this segment
            # This is a fallback. Ideally, chunks are long enough for the configured fade.
            # print(f"Warning: Audio segment (len {segment_len}) is too short for configured fade length ({self.fade_length_out_samples}). Using shorter fade ({actual_fade_len}).")
            # current_fade_in_window = np.linspace(0.0, 1.0, actual_fade_len, dtype=np.float32)
            # current_fade_out_window = np.linspace(1.0, 0.0, actual_fade_len, dtype=np.float32)
            
            # Or, slice the pre-calculated (e.g., Hanning) windows
            # This assumes pre-calculated windows are long enough. We already ensured fade_length_out_samples > 0.
            if self.fade_in_window.size >= actual_fade_len:
                 current_fade_in_window = self.fade_in_window[:actual_fade_len]
                 current_fade_out_window = self.fade_out_window[-actual_fade_len:] # Take from the end of the original fade-out
            else: # Should not happen if logic is correct, fallback to no fade for this short part
                return audio_segment


        faded_segment = audio_segment.copy() # Work on a copy

        # Apply fade-in
        faded_segment[:actual_fade_len] *= current_fade_in_window
        
        # Apply fade-out
        faded_segment[-actual_fade_len:] *= current_fade_out_window
        
        return faded_segment


    def get_base64_chunk(self, chunk: bytes) -> str:
        if not chunk:
             return ""
        audio_int16 = np.frombuffer(chunk, dtype=np.int16)
        if audio_int16.size == 0:
             return ""
        audio_float = audio_int16.astype(np.float32) / 32768.0
        downsampled_current_chunk_for_state = resample_poly(audio_float, self.output_fs, self.input_fs)

        part: np.ndarray # Explicitly type 'part'

        if self.previous_chunk is None:
            padded_audio_float = np.concatenate(
                (np.zeros(self.initial_padding_samples_in, dtype=np.float32), audio_float)
            )
            downsampled_padded_first_chunk = resample_poly(padded_audio_float, self.output_fs, self.input_fs)
            padding_samples_out = self.initial_padding_samples_in * self.output_fs // self.input_fs
            clean_downsampled_first_chunk = downsampled_padded_first_chunk[padding_samples_out:]
            
            if clean_downsampled_first_chunk.size == 0: # Handle very short first chunk
                self.previous_chunk = audio_float
                self.resampled_previous_chunk = downsampled_current_chunk_for_state
                return ""

            overlap_len_out = len(clean_downsampled_first_chunk) // 2
            part = clean_downsampled_first_chunk[:overlap_len_out]
            
            # For the very first part, only apply fade-out if it's not the only chunk expected
            # However, for consistency and to avoid complex logic about "is this the absolute end of stream?",
            # apply both. If it IS the absolute end, flush will handle its part.
            # A more sophisticated approach might avoid fade-out on the first chunk if it's also the last.
            # But we don't know that here.
            
        else:
            combined = np.concatenate((self.previous_chunk, audio_float))
            up_combined_resampled = resample_poly(combined, self.output_fs, self.input_fs)
            assert self.resampled_previous_chunk is not None
            prev_resampled_isolated_len = len(self.resampled_previous_chunk)
            
            if prev_resampled_isolated_len == 0: # Previous chunk was tiny
                # Treat current as if it's the first in this context (for slicing `up_combined_resampled`)
                # This means the part extracted should mostly be from `audio_float`'s contribution
                current_chunk_contrib_len_in_up = len(up_combined_resampled)
                h_prev_eff = 0
                h_cur_eff = current_chunk_contrib_len_in_up // 2
                part = up_combined_resampled[h_prev_eff:h_cur_eff]
            else:
                h_prev = prev_resampled_isolated_len // 2
                current_chunk_contrib_len_in_up = len(up_combined_resampled) - prev_resampled_isolated_len
                if current_chunk_contrib_len_in_up < 0: current_chunk_contrib_len_in_up = 0 # Safety
                h_cur = prev_resampled_isolated_len + (current_chunk_contrib_len_in_up // 2)
                part = up_combined_resampled[h_prev:h_cur]

        self.previous_chunk = audio_float
        self.resampled_previous_chunk = downsampled_current_chunk_for_state

        if part.size == 0:
            return ""

        # --- Apply fade here ---
        part_faded = self._apply_fade(part)
        # --- End Apply fade ---

        pcm_part = (part_faded * 32767).astype(np.int16).tobytes()
        ulaw_data = audioop.lin2ulaw(pcm_part, 2)
        return base64.b64encode(ulaw_data).decode('utf-8')

    def flush_base64_chunk(self) -> Optional[str]:
        if self.resampled_previous_chunk is not None and self.resampled_previous_chunk.size > 0:
            overlap_len_out = len(self.resampled_previous_chunk) // 2
            final_part_to_output_raw = np.array([], dtype=self.resampled_previous_chunk.dtype)

            # Determine the raw part to output
            if overlap_len_out == 0 and len(self.resampled_previous_chunk) > 0 :
                final_part_to_output_raw = self.resampled_previous_chunk[overlap_len_out:]
            elif overlap_len_out > 0 and overlap_len_out < len(self.resampled_previous_chunk):
                 final_part_to_output_raw = self.resampled_previous_chunk[overlap_len_out:]
            # else: final_part_to_output_raw remains empty

            if final_part_to_output_raw.size == 0:
                self.previous_chunk = None
                self.resampled_previous_chunk = None
                return None

            # --- Apply fade to the flushed part ---
            # For the flushed part, which is the absolute end, we typically only want a fade-out.
            # However, our _apply_fade does both. A more refined _apply_fade could take flags.
            # For now, applying both is okay; the fade-in at the start of this final segment
            # should ideally smoothly connect to the fade-out of the previous segment.
            final_part_faded = self._apply_fade(final_part_to_output_raw)
            # --- End Apply fade ---

            pcm_flush = (final_part_faded * 32767).astype(np.int16).tobytes()
            self.previous_chunk = None
            self.resampled_previous_chunk = None
            ulaw_data = audioop.lin2ulaw(pcm_flush, 2)
            return base64.b64encode(ulaw_data).decode('utf-8')

        self.previous_chunk = None
        self.resampled_previous_chunk = None
        return None