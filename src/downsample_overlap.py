import base64
import numpy as np
from scipy.signal import resample_poly
import audioop # For µ-law conversion
from typing import Optional, Tuple

class ResampleOverlapUlaw:
    """
    Manages chunk-wise audio resampling (e.g., 24kHz to 8kHz) and µ-law encoding
    with overlap-save handling to mitigate boundary artifacts.

    This class processes sequential audio chunks (assumed 16-bit PCM),
    resamples them, converts them to 8-bit µ-law, and manages overlap
    between chunks. The processed, resampled, µ-law encoded audio segments
    are returned as Base64 encoded strings. It maintains internal state to
    handle the overlap correctly across calls.
    """
    INPUT_SR = 24000
    OUTPUT_SR = 8000
    # Overlap duration in milliseconds. This should be sufficient to cover
    # the transient response of the resample_poly filter. 20-40ms is common.
    OVERLAP_MS = 40

    # Calculate overlap samples at input and output rates
    # These are float values initially, then cast to int.
    _OVERLAP_SAMPLES_IN_FLOAT = INPUT_SR * OVERLAP_MS / 1000.0
    OVERLAP_SAMPLES_IN = int(_OVERLAP_SAMPLES_IN_FLOAT)

    # The number of samples to discard from the beginning of the resampled output.
    # This corresponds to the OVERLAP_SAMPLES_IN prepended before resampling.
    # Must be calculated carefully to match the resampling ratio.
    # For resample_poly, the output length is floor(input_len * up / down).
    # So, the length of the resampled overlap is floor(OVERLAP_SAMPLES_IN * OUTPUT_SR / INPUT_SR).
    # However, to be robust, we resample a block of (OVERLAP_SAMPLES_IN + CHUNK_SAMPLES_IN),
    # and then discard the portion corresponding to OVERLAP_SAMPLES_IN.
    # The length of the resampled version of OVERLAP_SAMPLES_IN is what we need to discard.
    # For FIR filters in resample_poly, the group delay is roughly half the filter length.
    # A simpler way for overlap-save is to make the discard length:
    OVERLAP_SAMPLES_OUT = int(OUTPUT_SR * OVERLAP_MS / 1000.0)


    def __init__(self):
        """
        Initializes the ResampleOverlapUlaw processor.
        Sets up internal state for overlap handling.
        """
        # Stores the tail of the *previous input chunk* (24kHz float32) to be
        # prepended to the current chunk for overlap-save.
        self.input_overlap_buffer: Optional[np.ndarray] = None
        self.is_first_chunk: bool = True

    def _pcm_float_to_ulaw_bytes(self, audio_float: np.ndarray) -> bytes:
        """Converts float32 PCM audio (-1.0 to 1.0) to µ-law bytes."""
        if audio_float.size == 0:
            return b""
        # Convert float32 to int16
        audio_int16 = (audio_float * 32767.0).astype(np.int16)
        # Convert int16 numpy array to bytes
        pcm_bytes = audio_int16.tobytes()
        # Convert linear PCM bytes to µ-law bytes
        # The '2' indicates 2 bytes per sample for the input pcm_bytes
        ulaw_audio_bytes = audioop.lin2ulaw(pcm_bytes, 2)
        return ulaw_audio_bytes

    def get_base64_chunk(self, chunk_bytes: bytes) -> str:
        """
        Processes an incoming audio chunk, resamples it to 8kHz, converts to
        µ-law, and returns the relevant segment as Base64.

        Args:
            chunk_bytes: Raw audio data bytes (PCM 16-bit signed integer format, 24kHz).

        Returns:
            A Base64 encoded string representing the resampled, µ-law encoded
            audio segment. Returns an empty string if the input chunk is empty
            and results in no processable audio.
        """
        if not chunk_bytes:
            # If the input chunk is empty, there's nothing to process from it.
            # If there was a previous overlap buffer, it should be flushed separately.
            return ""

        audio_int16 = np.frombuffer(chunk_bytes, dtype=np.int16)
        if audio_int16.size == 0:
            return ""

        current_input_float = audio_int16.astype(np.float32) / 32768.0

        # Prepare the block for resampling by prepending overlap
        if self.is_first_chunk:
            # For the first chunk, prepend zeros to stabilize the filter.
            # The length of zeros should match OVERLAP_SAMPLES_IN.
            prepend_data = np.zeros(self.OVERLAP_SAMPLES_IN, dtype=np.float32)
            block_to_resample = np.concatenate((prepend_data, current_input_float))
            self.is_first_chunk = False
        elif self.input_overlap_buffer is not None:
            block_to_resample = np.concatenate((self.input_overlap_buffer, current_input_float))
        else:
            # Should not happen after the first chunk if input_overlap_buffer is managed correctly
            # For safety, behave like the first chunk if input_overlap_buffer is somehow None
            prepend_data = np.zeros(self.OVERLAP_SAMPLES_IN, dtype=np.float32)
            block_to_resample = np.concatenate((prepend_data, current_input_float))


        # Resample (e.g., 24kHz to 8kHz)
        # `up` is OUTPUT_SR, `down` is INPUT_SR. Simplified: up=1, down=3 for 8k from 24k
        resampled_float = resample_poly(block_to_resample, self.OUTPUT_SR, self.INPUT_SR)

        # Discard the initial part of the resampled block that corresponds to the overlap
        # This is the "save" part of overlap-save.
        # The number of samples to discard is OVERLAP_SAMPLES_OUT.
        if resampled_float.size > self.OVERLAP_SAMPLES_OUT:
            output_segment_float = resampled_float[self.OVERLAP_SAMPLES_OUT:]
        else:
            # This can happen if the input chunk + overlap is very small,
            # resulting in a resampled block shorter than the discard length.
            # Output nothing in this case from this chunk, and carry over the input.
            output_segment_float = np.array([], dtype=np.float32)

        # Save the tail of the *current input chunk* (at 24kHz) for the next iteration's overlap
        if current_input_float.size >= self.OVERLAP_SAMPLES_IN:
            self.input_overlap_buffer = current_input_float[-self.OVERLAP_SAMPLES_IN:]
        else:
            # If current chunk is shorter than overlap, the whole chunk becomes overlap
            self.input_overlap_buffer = current_input_float.copy() # Use copy

        if output_segment_float.size == 0:
            return ""

        # Convert the valid output segment to µ-law bytes
        ulaw_bytes = self._pcm_float_to_ulaw_bytes(output_segment_float)

        return base64.b64encode(ulaw_bytes).decode('utf-8')

    def flush_base64_chunk(self) -> str:
        """
        Processes any remaining audio in the internal overlap buffer, resamples
        it, converts to µ-law, and returns as Base64. This should be called
        once after all input chunks have been passed to get_base64_chunk.

        Returns:
            A Base64 encoded string for the final audio segment, or an empty
            string if there's nothing to flush.
        """
        output_str = ""
        if self.input_overlap_buffer is not None and self.input_overlap_buffer.size > 0:
            # Treat the remaining overlap buffer as the final piece of audio.
            # To properly resample this final segment without cutting off filter effects,
            # we can conceptually think of it as a chunk prepended by zeros (as it's the end).
            # Or, if resample_poly handles edges well for short inputs, just resample it.
            # For overlap-save, the self.input_overlap_buffer is the part that hasn't generated
            # its corresponding output yet. We resample it directly.
            
            # To ensure filter stabilization for this last bit, we can prepend zeros
            # and then take the relevant part.
            prepend_zeros = np.zeros(self.OVERLAP_SAMPLES_IN, dtype=np.float32)
            block_to_resample = np.concatenate((prepend_zeros, self.input_overlap_buffer))
            
            resampled_float = resample_poly(block_to_resample, self.OUTPUT_SR, self.INPUT_SR)
            
            # We discard the part corresponding to prepend_zeros
            if resampled_float.size > self.OVERLAP_SAMPLES_OUT:
                final_segment_float = resampled_float[self.OVERLAP_SAMPLES_OUT:]
            else:
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
    processor = ResampleOverlapUlaw()

    # Simulate incoming audio chunks (24kHz, 16-bit PCM)
    # Each sample is 2 bytes. Chunk size in samples.
    # Let's say 100ms chunks: 0.1 * 24000 = 2400 samples = 4800 bytes
    chunk_samples = 2400
    num_chunks = 5

    print(f"Input SR: {processor.INPUT_SR} Hz, Output SR: {processor.OUTPUT_SR} Hz (µ-law)")
    print(f"Overlap: {processor.OVERLAP_MS} ms")
    print(f"Input Overlap Samples (24kHz): {processor.OVERLAP_SAMPLES_IN}")
    print(f"Output Discard Samples (8kHz): {processor.OVERLAP_SAMPLES_OUT}")
    print("-" * 30)

    all_output_ulaw_bytes = b""

    for i in range(num_chunks):
        # Create a dummy audio chunk (e.g., a sine wave segment)
        # Make frequency change per chunk to hear transitions
        frequency = 440 + i * 100
        t = np.arange(chunk_samples) / processor.INPUT_SR
        # Ensure signal is within int16 range for realistic input
        signal_chunk_float = 0.5 * np.sin(2 * np.pi * frequency * t)
        signal_chunk_int16 = (signal_chunk_float * 32767).astype(np.int16)
        chunk_bytes = signal_chunk_int16.tobytes()

        print(f"Processing chunk {i+1}/{num_chunks} (input: {len(chunk_bytes)} bytes)")
        base64_output = processor.get_base64_chunk(chunk_bytes)
        
        if base64_output:
            decoded_ulaw_bytes = base64.b64decode(base64_output)
            all_output_ulaw_bytes += decoded_ulaw_bytes
            print(f" -> Output: {len(base64_output)} b64 chars ({len(decoded_ulaw_bytes)} µ-law bytes)")
        else:
            print(f" -> Output: Empty")

    print("-" * 30)
    print("Flushing remaining audio...")
    base64_flush_output = processor.flush_base64_chunk()
    if base64_flush_output:
        decoded_ulaw_bytes = base64.b64decode(base64_flush_output)
        all_output_ulaw_bytes += decoded_ulaw_bytes
        print(f" -> Flushed output: {len(base64_flush_output)} b64 chars ({len(decoded_ulaw_bytes)} µ-law bytes)")
    else:
        print(f" -> Flushed output: Empty")

    print("-" * 30)
    print(f"Total µ-law bytes generated: {len(all_output_ulaw_bytes)}")

    # To verify, you might save `all_output_ulaw_bytes` to a .ul file
    # or convert it back to PCM and listen.
    # For example, saving to a raw µ-law file:
    if all_output_ulaw_bytes:
        with open("output_8k.ulaw", "wb") as f:
            f.write(all_output_ulaw_bytes)
        print("Saved total output to output_8k.ulaw")
        print("You can play it with SoX: play -r 8000 -e u-law -b 8 -c 1 output_8k.ulaw")

        # Optional: Convert back to PCM for analysis if needed
        # pcm_data_bytes = audioop.ulaw2lin(all_output_ulaw_bytes, 2)
        # pcm_data_int16 = np.frombuffer(pcm_data_bytes, dtype=np.int16)
        # from scipy.io.wavfile import write as wav_write
        # wav_write("output_8k_reverted_pcm.wav", processor.OUTPUT_SR, pcm_data_int16)
        # print("Saved reverted PCM to output_8k_reverted_pcm.wav")