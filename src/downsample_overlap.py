import base64
import numpy as np
from scipy.signal import resample_poly
import audioop # For µ-law conversion
from typing import Optional, Tuple

class ResampleOverlapUlaw:
    """
    Manages chunk-wise audio resampling (e.g., 24kHz to 8kHz) and µ-law encoding
    with an optimized overlap-save strategy to mitigate boundary artifacts and noise.

    This version uses a longer overlap and a specific Kaiser window for resampling
    to improve smoothness and filter performance.
    """
    INPUT_SR = 24000
    OUTPUT_SR = 8000

    # Increased overlap duration in milliseconds.
    # A longer overlap provides more context for the filter, reducing edge artifacts.
    OVERLAP_MS = 60 # Increased from 30ms, can be tuned further (e.g., 40, 80)

    # Kaiser window parameters for resample_poly.
    # Beta > 5 gives more stopband attenuation. Beta=8.0 is a common choice for good attenuation.
    # This can help reduce aliasing artifacts if the default filter was insufficient.
    KAISER_WINDOW_PARAMS = ('kaiser', 8.0) # Default for resample_poly is ('kaiser', 5.0)

    # Calculate overlap samples at input and output rates
    # Ensure these are integers
    OVERLAP_SAMPLES_IN = int(INPUT_SR * OVERLAP_MS / 1000.0)

    # The number of samples to discard from the beginning of the resampled output.
    # This should correspond to the length of the resampled version of OVERLAP_SAMPLES_IN.
    # The output length of resample_poly is floor(input_len * up / down).
    # So, the length of the resampled overlap to discard is:
    # floor(OVERLAP_SAMPLES_IN * OUTPUT_SR / INPUT_SR)
    # However, to be robust and ensure the transient is fully covered,
    # using ceil(OUTPUT_SR * OVERLAP_MS / 1000.0) is often safer for the discard length.
    OVERLAP_SAMPLES_OUT = int(np.ceil(OUTPUT_SR * OVERLAP_MS / 1000.0))
    # Alternative strict calculation based on resample_poly behavior:
    # OVERLAP_SAMPLES_OUT = int(np.floor(OVERLAP_SAMPLES_IN * OUTPUT_SR / INPUT_SR))


    def __init__(self):
        """
        Initializes the ResampleOverlapUlawOptimized processor.
        """
        self.input_overlap_buffer: Optional[np.ndarray] = None
        self.is_first_chunk: bool = True
        if self.OVERLAP_SAMPLES_IN == 0 and self.OVERLAP_MS > 0:
            print(f"Warning: OVERLAP_SAMPLES_IN is 0 with OVERLAP_MS={self.OVERLAP_MS}. "
                  f"Input SR might be too low or OVERLAP_MS too small for effective sample count.")
        elif self.OVERLAP_SAMPLES_OUT == 0 and self.OVERLAP_MS > 0:
             print(f"Warning: OVERLAP_SAMPLES_OUT is 0 with OVERLAP_MS={self.OVERLAP_MS}. "
                  f"Output SR might be too low or OVERLAP_MS too small for effective sample count.")


    def _pcm_float_to_ulaw_bytes(self, audio_float: np.ndarray) -> bytes:
        """Converts float32 PCM audio (-1.0 to 1.0) to µ-law bytes."""
        if audio_float.size == 0:
            return b""
        # Clip to ensure values are within int16 range before conversion
        audio_int16 = np.clip(audio_float * 32768.0, -32768.0, 32767.0).astype(np.int16)
        pcm_bytes = audio_int16.tobytes()
        ulaw_audio_bytes = audioop.lin2ulaw(pcm_bytes, 2)
        return ulaw_audio_bytes

    def get_base64_chunk(self, chunk_bytes: bytes) -> str:
        """
        Processes an incoming audio chunk, resamples it to 8kHz, converts to
        µ-law, and returns the relevant segment as Base64.
        """
        if not chunk_bytes:
            return ""

        audio_int16 = np.frombuffer(chunk_bytes, dtype=np.int16)
        if audio_int16.size == 0:
            return ""

        current_input_float = audio_int16.astype(np.float32) / 32768.0

        # Prepare block for resampling by prepending overlap
        if self.is_first_chunk:
            if self.OVERLAP_SAMPLES_IN > 0:
                prepend_data = np.zeros(self.OVERLAP_SAMPLES_IN, dtype=np.float32)
                block_to_resample = np.concatenate((prepend_data, current_input_float))
            else: # No overlap configured
                block_to_resample = current_input_float
            self.is_first_chunk = False
        elif self.input_overlap_buffer is not None and self.input_overlap_buffer.size > 0:
            # Prepend the saved overlap from the previous chunk
            block_to_resample = np.concatenate((self.input_overlap_buffer, current_input_float))
        else: # No overlap buffer available (e.g. if OVERLAP_SAMPLES_IN is 0 or it was an empty chunk before)
            if self.OVERLAP_SAMPLES_IN > 0: # Still attempt to prepend zeros if overlap is configured
                prepend_data = np.zeros(self.OVERLAP_SAMPLES_IN, dtype=np.float32)
                block_to_resample = np.concatenate((prepend_data, current_input_float))
            else:
                block_to_resample = current_input_float


        # Resample using the specified Kaiser window
        resampled_float = resample_poly(
            block_to_resample,
            self.OUTPUT_SR,
            self.INPUT_SR,
            window=self.KAISER_WINDOW_PARAMS
        )

        # Discard the initial part of the resampled block (the "save" part of overlap-save)
        if self.OVERLAP_SAMPLES_OUT > 0 and resampled_float.size > self.OVERLAP_SAMPLES_OUT:
            output_segment_float = resampled_float[self.OVERLAP_SAMPLES_OUT:]
        elif self.OVERLAP_SAMPLES_OUT == 0 : # No output overlap discard configured
             output_segment_float = resampled_float
        else:
            # Resampled block is too short to discard the full overlap.
            # This can happen if (input_chunk + overlap_buffer) is very small.
            # Outputting nothing is safer to avoid artifacts.
            output_segment_float = np.array([], dtype=np.float32)

        # Save the tail of the *current input chunk* for the next iteration's overlap
        if self.OVERLAP_SAMPLES_IN > 0:
            if current_input_float.size >= self.OVERLAP_SAMPLES_IN:
                self.input_overlap_buffer = current_input_float[-self.OVERLAP_SAMPLES_IN:]
            else:
                # If current chunk is shorter than overlap, pad it with leading zeros
                # to make its length OVERLAP_SAMPLES_IN for the buffer.
                # This ensures the next chunk always gets a consistently sized overlap.
                # Alternatively, just save what's available:
                self.input_overlap_buffer = current_input_float.copy()
                # If a consistent overlap buffer length is strictly needed for some reason:
                # padding_zeros = np.zeros(self.OVERLAP_SAMPLES_IN - current_input_float.size, dtype=np.float32)
                # self.input_overlap_buffer = np.concatenate((padding_zeros, current_input_float))
        else: # No overlap configured
            self.input_overlap_buffer = None


        if output_segment_float.size == 0:
            return ""

        ulaw_bytes = self._pcm_float_to_ulaw_bytes(output_segment_float)
        return base64.b64encode(ulaw_bytes).decode('utf-8')

    def flush_base64_chunk(self) -> str:
        """
        Processes any remaining audio in the internal overlap buffer.
        """
        output_str = ""
        if self.input_overlap_buffer is not None and self.input_overlap_buffer.size > 0:
            # This is the final segment of actual input audio.
            # To resample it correctly, prepend zeros for filter stabilization,
            # then discard the resampled portion corresponding to these leading zeros.

            if self.OVERLAP_SAMPLES_IN > 0:
                prepend_zeros = np.zeros(self.OVERLAP_SAMPLES_IN, dtype=np.float32)
                # The block for resampling is the leading zeros + the final audio buffer
                block_to_resample = np.concatenate((prepend_zeros, self.input_overlap_buffer))
            else: # No overlap configured
                 block_to_resample = self.input_overlap_buffer


            resampled_float = resample_poly(
                block_to_resample,
                self.OUTPUT_SR,
                self.INPUT_SR,
                window=self.KAISER_WINDOW_PARAMS
            )

            # Discard the part corresponding to the prepended zeros
            if self.OVERLAP_SAMPLES_OUT > 0 and resampled_float.size > self.OVERLAP_SAMPLES_OUT:
                final_segment_float = resampled_float[self.OVERLAP_SAMPLES_OUT:]
            elif self.OVERLAP_SAMPLES_OUT == 0: # No output overlap discard
                final_segment_float = resampled_float
            else: # Resampled output too short
                final_segment_float = np.array([], dtype=np.float32)


            if final_segment_float.size > 0:
                ulaw_bytes = self._pcm_float_to_ulaw_bytes(final_segment_float)
                output_str = base64.b64encode(ulaw_bytes).decode('utf-8')

        # Clear state
        self.input_overlap_buffer = None
        self.is_first_chunk = True
        return output_str

# --- Example Usage (for testing) ---
if __name__ == '__main__':
    processor = ResampleOverlapUlawOptimized()

    chunk_duration_ms = 100 # Duration of each input audio chunk
    chunk_samples = int(processor.INPUT_SR * chunk_duration_ms / 1000.0)
    num_chunks = 5

    print(f"Input SR: {processor.INPUT_SR} Hz, Output SR: {processor.OUTPUT_SR} Hz (µ-law)")
    print(f"Overlap: {processor.OVERLAP_MS} ms, Kaiser Window: {processor.KAISER_WINDOW_PARAMS}")
    print(f"Input Overlap Samples (at {processor.INPUT_SR}Hz): {processor.OVERLAP_SAMPLES_IN}")
    print(f"Output Discard Samples (at {processor.OUTPUT_SR}Hz): {processor.OVERLAP_SAMPLES_OUT}")
    print(f"Input chunk samples: {chunk_samples} ({chunk_duration_ms}ms)")
    print("-" * 30)

    all_output_ulaw_bytes = b""
    total_input_samples_processed = 0 # For generating continuous test signal

    for i in range(num_chunks):
        # Create a dummy audio chunk (e.g., a sine wave segment with continuous phase)
        frequency = 440 + i * 110 # Vary frequency to test transitions
        
        # Calculate time array for this chunk ensuring phase continuity
        t_start_offset = total_input_samples_processed / processor.INPUT_SR
        t_chunk_relative = np.arange(chunk_samples) / processor.INPUT_SR
        t_absolute = t_start_offset + t_chunk_relative
        
        signal_chunk_float = 0.6 * np.sin(2 * np.pi * frequency * t_absolute) # Amplitude 0.6
        signal_chunk_int16 = (signal_chunk_float * 32767).astype(np.int16)
        chunk_bytes = signal_chunk_int16.tobytes()
        
        total_input_samples_processed += chunk_samples

        print(f"Processing chunk {i+1}/{num_chunks} (input: {len(chunk_bytes)} bytes, {chunk_samples} samples)")
        base64_output = processor.get_base64_chunk(chunk_bytes)
        
        if base64_output:
            decoded_ulaw_bytes = base64.b64decode(base64_output)
            all_output_ulaw_bytes += decoded_ulaw_bytes
            # Expected output samples from this chunk (main part)
            expected_output_samples_chunk = int(np.floor(chunk_samples * processor.OUTPUT_SR / processor.INPUT_SR))
            print(f" -> Output: {len(base64_output)} b64 chars ({len(decoded_ulaw_bytes)} µ-law bytes, "
                  f"expected ~{expected_output_samples_chunk} samples from current chunk's main part)")
        else:
            print(f" -> Output: Empty")

    print("-" * 30)
    print("Flushing remaining audio...")
    base64_flush_output = processor.flush_base64_chunk()
    if base64_flush_output:
        decoded_ulaw_bytes = base64.b64decode(base64_flush_output)
        all_output_ulaw_bytes += decoded_ulaw_bytes
        # Expected output samples from flush (corresponds to OVERLAP_SAMPLES_IN)
        expected_output_samples_flush = 0
        if processor.OVERLAP_SAMPLES_IN > 0 :
             expected_output_samples_flush = int(np.floor(processor.OVERLAP_SAMPLES_IN * processor.OUTPUT_SR / processor.INPUT_SR))
        print(f" -> Flushed output: {len(base64_flush_output)} b64 chars ({len(decoded_ulaw_bytes)} µ-law bytes, "
              f"expected ~{expected_output_samples_flush} samples from final overlap)")
    else:
        print(f" -> Flushed output: Empty")

    print("-" * 30)
    print(f"Total µ-law bytes generated: {len(all_output_ulaw_bytes)}")
    # Calculate total expected output samples based on total input samples
    expected_total_output_samples = int(np.floor(total_input_samples_processed * processor.OUTPUT_SR / processor.INPUT_SR))
    print(f"Expected total µ-law samples (approx): {expected_total_output_samples}")


    if all_output_ulaw_bytes:
        filename_ulaw = "output_8k_optimized.ulaw"
        filename_wav = "output_8k_optimized_reverted_pcm.wav"
        with open(filename_ulaw, "wb") as f:
            f.write(all_output_ulaw_bytes)
        print(f"Saved total output to {filename_ulaw}")
        print(f"You can play it with SoX: play -r {processor.OUTPUT_SR} -e u-law -b 8 -c 1 {filename_ulaw}")

        # Optional: Convert back to PCM for analysis if needed
        # pcm_data_bytes = audioop.ulaw2lin(all_output_ulaw_bytes, 2)
        # pcm_data_int16 = np.frombuffer(pcm_data_bytes, dtype=np.int16)
        # from scipy.io.wavfile import write as wav_write
        # wav_write(filename_wav, processor.OUTPUT_SR, pcm_data_int16)
        # print(f"Saved reverted PCM to {filename_wav}")