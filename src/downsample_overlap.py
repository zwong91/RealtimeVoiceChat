import base64
import numpy as np
from scipy.signal import resample_poly
import audioop # For µ-law conversion
from typing import Optional

class ResampleOverlapUlaw:
    """
    管理块式音频重采样 (24kHz -> 8kHz) 和 µ-law 编码，
    使用Overlap-Save策略减少边界伪影。
    此版本旨在简化和稳健性。
    """
    INPUT_SR = 24000
    OUTPUT_SR = 8000

    # 重叠时长 (毫秒)
    # 40ms 是一个常用值，在伪影减少和延迟之间取得平衡。
    OVERLAP_MS = 30

    # 根据输入采样率计算重叠样本数
    # 例如: 24000 Hz * 40 ms / 1000 = 960 样本
    OVERLAP_SAMPLES_IN = int(INPUT_SR * OVERLAP_MS / 1000.0)

    # 计算从重采样后输出中需要丢弃的样本数
    # 这应该精确对应 OVERLAP_SAMPLES_IN 重采样后的长度。
    # 对于 24kHz -> 8kHz (系数 1/3), OVERLAP_SAMPLES_IN = 960,
    # OVERLAP_SAMPLES_OUT = 960 * (8000/24000) = 960 / 3 = 320 样本。
    # 注意: resample_poly 输出长度是 floor(输入长度 * up / down)。
    # 由于此例中 OVERLAP_SAMPLES_IN (960) 是 (INPUT_SR/OUTPUT_SR)=3 的倍数，floor 不是严格必需的。
    OVERLAP_SAMPLES_OUT = int(OVERLAP_SAMPLES_IN * OUTPUT_SR / INPUT_SR)

    def __init__(self):
        """
        初始化处理器。
        """
        # 存储上一个24kHz float输入块的尾部，用于下一次的重叠
        self.input_overlap_buffer: Optional[np.ndarray] = None
        self.is_first_chunk: bool = True

        if self.OVERLAP_SAMPLES_IN <= 0 or self.OVERLAP_SAMPLES_OUT <= 0:
            print(f"警告: 重叠样本数非正。 "
                  f"输入重叠: {self.OVERLAP_SAMPLES_IN}, 输出丢弃: {self.OVERLAP_SAMPLES_OUT}."
                  f"Overlap-Save 可能无法正常工作。")
            # 如果重叠是必须的，可以考虑在此处抛出错误

    def _pcm_float_to_ulaw_bytes(self, audio_float: np.ndarray) -> bytes:
        """将 float32 PCM 音频 (-1.0 到 1.0) 转换为 µ-law 字节串。"""
        if audio_float.size == 0:
            return b""
        # 缩放到 int16 范围。使用 32767.0 作为乘数。
        # np.clip 确保值严格在 int16 限制内，防止溢出。
        audio_int16 = np.clip(audio_float * 32767.0, -32768.0, 32767.0).astype(np.int16)
        # 将 int16 numpy 数组转换为字节串
        pcm_bytes = audio_int16.tobytes()
        # 将线性 PCM 字节串转换为 µ-law 字节串 (输入是2字节样本)
        ulaw_audio_bytes = audioop.lin2ulaw(pcm_bytes, 2)
        return ulaw_audio_bytes

    def get_base64_chunk(self, chunk_bytes: bytes) -> str:
        """
        处理输入的音频块，重采样到8kHz，转换为µ-law，并返回Base64编码的有效片段。
        """
        if not chunk_bytes:
            return "" # 空输入块

        audio_int16 = np.frombuffer(chunk_bytes, dtype=np.int16)
        if audio_int16.size == 0:
            return "" # 输入块解析后为空

        # 将16位整数PCM转换为float32并归一化到 [-1.0, 1.0)
        current_input_float = audio_int16.astype(np.float32) / 32768.0

        # 准备用于重采样的块 (前置重叠部分)
        if self.is_first_chunk:
            # 对于第一个块，在其前面填充静音（0）以稳定滤波器
            prepend_data = np.zeros(self.OVERLAP_SAMPLES_IN, dtype=np.float32)
            block_to_resample = np.concatenate((prepend_data, current_input_float))
            self.is_first_chunk = False
        elif self.input_overlap_buffer is not None:
            # 对于后续块，前置上一个输入块保存的尾部重叠数据
            block_to_resample = np.concatenate((self.input_overlap_buffer, current_input_float))
        else:
            # 此情况理论上在第一个块之后不应发生 (如果 OVERLAP_SAMPLES_IN > 0)。
            # 这表示 input_overlap_buffer 因某种原因变回 None。作为后备，填充静音。
            prepend_data = np.zeros(self.OVERLAP_SAMPLES_IN, dtype=np.float32)
            block_to_resample = np.concatenate((prepend_data, current_input_float))
        
        # 重采样 (例如 24kHz -> 8kHz)。使用 resample_poly 的默认窗函数。
        resampled_float = resample_poly(
            block_to_resample,
            self.OUTPUT_SR, # up (目标采样率)
            self.INPUT_SR   # down (原始采样率)
        )
        # 预期长度: floor(len(block_to_resample) * OUTPUT_SR / INPUT_SR)

        # 丢弃重采样后块的初始部分 (Overlap-Save 中的 "Save" 步骤)。
        # 这部分对应于之前添加的（静音或实际的）重叠数据。
        if resampled_float.size > self.OVERLAP_SAMPLES_OUT:
            output_segment_float = resampled_float[self.OVERLAP_SAMPLES_OUT:]
        else:
            # 如果重采样后的块长度不大于要丢弃的长度，则此块不产生有效输出。
            # 这可能在 (输入块 + 重叠缓冲) 非常短，或者 OVERLAP_SAMPLES_OUT 计算错误/过大时发生。
            output_segment_float = np.array([], dtype=np.float32)

        # 保存当前输入块 (24kHz float) 的尾部，用于下一次迭代的重叠。
        if current_input_float.size >= self.OVERLAP_SAMPLES_IN:
            self.input_overlap_buffer = current_input_float[-self.OVERLAP_SAMPLES_IN:]
        else:
            # 如果当前块比期望的重叠长度还短，则整个当前块成为重叠缓冲。
            self.input_overlap_buffer = current_input_float.copy()
        
        if output_segment_float.size == 0:
            return ""

        # 将有效的输出片段转换为 µ-law 字节并进行 Base64 编码
        ulaw_bytes = self._pcm_float_to_ulaw_bytes(output_segment_float)
        return base64.b64encode(ulaw_bytes).decode('utf-8')

    def flush_base64_chunk(self) -> str:
        """
        处理 input_overlap_buffer 中剩余的音频。
        这个缓冲中保存的是最后一个实际输入块的尾部。
        应该在所有 get_base64_chunk 调用完毕后调用一次。
        """
        output_str = ""
        if self.input_overlap_buffer is not None and self.input_overlap_buffer.size > 0:
            # self.input_overlap_buffer 包含原始信号的最后一部分。
            # 为清晰地重采样它，我们将其视为一个新块，并在其前面填充静音
            # 以使滤波器稳定，然后丢弃重采样后的静音部分。
            
            prepend_zeros = np.zeros(self.OVERLAP_SAMPLES_IN, dtype=np.float32)
            # 用于重采样的块是：前导静音 + 缓冲中最后的音频数据
            block_to_resample = np.concatenate((prepend_zeros, self.input_overlap_buffer))
            
            resampled_float = resample_poly(
                block_to_resample,
                self.OUTPUT_SR,
                self.INPUT_SR
            )

            # 丢弃对应于前导静音的部分
            if resampled_float.size > self.OVERLAP_SAMPLES_OUT:
                final_segment_float = resampled_float[self.OVERLAP_SAMPLES_OUT:]
            else:
                final_segment_float = np.array([], dtype=np.float32)

            if final_segment_float.size > 0:
                ulaw_bytes = self._pcm_float_to_ulaw_bytes(final_segment_float)
                output_str = base64.b64encode(ulaw_bytes).decode('utf-8')
        
        # 清理状态，为下一次可能的序列处理做准备
        self.input_overlap_buffer = None
        self.is_first_chunk = True
        return output_str

# --- 示例用法 (用于测试) ---
if __name__ == '__main__':
    processor = ResampleOverlapUlaw()

    # 每个输入音频块的持续时间
    chunk_duration_ms = 100 # 例如 100ms
    # 根据输入采样率计算每个块的样本数
    chunk_samples = int(processor.INPUT_SR * chunk_duration_ms / 1000.0)
    num_chunks = 5 # 模拟处理5个块

    print(f"输入采样率: {processor.INPUT_SR} Hz, 输出采样率: {processor.OUTPUT_SR} Hz (µ-law)")
    print(f"重叠: {processor.OVERLAP_MS} ms")
    print(f"输入重叠样本数 (在 {processor.INPUT_SR}Hz): {processor.OVERLAP_SAMPLES_IN}")
    print(f"输出丢弃样本数 (在 {processor.OUTPUT_SR}Hz): {processor.OVERLAP_SAMPLES_OUT}")
    print(f"每个输入块的样本数: {chunk_samples} ({chunk_duration_ms}ms)")
    print("-" * 30)

    all_output_ulaw_bytes = b""
    total_input_samples_processed = 0 # 用于生成连续的测试信号

    for i in range(num_chunks):
        # 创建一个虚拟音频块 (例如，相位连续的正弦波段)
        frequency = 200 + i * 150 # 改变频率以测试过渡
        
        t_start_offset = total_input_samples_processed / processor.INPUT_SR
        t_chunk_relative = np.arange(chunk_samples) / processor.INPUT_SR
        t_absolute = t_start_offset + t_chunk_relative
        
        # 幅度设置为0.7，以避免削波，同时为µ-law提供足够的信号电平
        signal_chunk_float = 0.7 * np.sin(2 * np.pi * frequency * t_absolute) 
        signal_chunk_int16 = (signal_chunk_float * 32767.0).astype(np.int16) # 使用32767.0进行缩放
        chunk_bytes = signal_chunk_int16.tobytes()
        
        total_input_samples_processed += chunk_samples

        print(f"处理块 {i+1}/{num_chunks} (输入: {len(chunk_bytes)} 字节, {chunk_samples} 样本)")
        base64_output = processor.get_base64_chunk(chunk_bytes)
        
        if base64_output:
            decoded_ulaw_bytes = base64.b64decode(base64_output)
            all_output_ulaw_bytes += decoded_ulaw_bytes
            # 当前块主要部分预期输出的样本数
            # output_segment_float 的长度大约是 floor(chunk_samples * OUTPUT_SR / INPUT_SR)
            expected_output_samples_chunk = int(np.floor(chunk_samples * processor.OUTPUT_SR / processor.INPUT_SR))
            print(f" -> 输出: {len(base64_output)} Base64字符 ({len(decoded_ulaw_bytes)} µ-law字节, "
                  f"来自当前块主体的预期样本数约 {expected_output_samples_chunk})")
        else:
            print(f" -> 输出: 空")

    print("-" * 30)
    print("处理剩余音频 (flush)...")
    base64_flush_output = processor.flush_base64_chunk()
    if base64_flush_output:
        decoded_ulaw_bytes = base64.b64decode(base64_flush_output)
        all_output_ulaw_bytes += decoded_ulaw_bytes
        # flush 操作处理的是长度为 OVERLAP_SAMPLES_IN 的输入缓冲
        # 其预期输出样本数是 floor(OVERLAP_SAMPLES_IN * OUTPUT_SR / INPUT_SR) = OVERLAP_SAMPLES_OUT
        expected_output_samples_flush = processor.OVERLAP_SAMPLES_OUT
        print(f" -> Flush输出: {len(base64_flush_output)} Base64字符 ({len(decoded_ulaw_bytes)} µ-law字节, "
              f"来自最后重叠的预期样本数约 {expected_output_samples_flush})")
    else:
        print(f" -> Flush输出: 空")

    print("-" * 30)
    print(f"总共生成的µ-law字节数: {len(all_output_ulaw_bytes)}")
    # 根据总输入样本数计算预期的总输出样本数
    # 这个计算忽略了resample_poly内部滤波器引入的非常微小的群延迟（通常几个样本）
    expected_total_output_samples = int(np.floor(total_input_samples_processed * processor.OUTPUT_SR / processor.INPUT_SR))
    print(f"预期的µ-law总样本数 (近似): {expected_total_output_samples}")
    print(f"实际生成的µ-law总样本数: {len(all_output_ulaw_bytes)}") # 注意这是字节数，每个µ-law样本1字节

    if all_output_ulaw_bytes:
        filename_ulaw = "output_8k_revised.ulaw"
        filename_wav = "output_8k_revised_reverted_pcm.wav" # 如果需要转换回PCM进行分析
        with open(filename_ulaw, "wb") as f:
            f.write(all_output_ulaw_bytes)
        print(f"总输出已保存到 {filename_ulaw}")
        print(f"你可以使用 SoX 播放: play -r {processor.OUTPUT_SR} -e u-law -b 8 -c 1 {filename_ulaw}")

        # 可选: 将 µ-law 转回 PCM 用于更详细的分析或试听 (需要scipy.io.wavfile)
        # try:
        #     from scipy.io.wavfile import write as wav_write
        #     pcm_data_bytes = audioop.ulaw2lin(all_output_ulaw_bytes, 2) # 2字节线性PCM
        #     pcm_data_int16 = np.frombuffer(pcm_data_bytes, dtype=np.int16)
        #     wav_write(filename_wav, processor.OUTPUT_SR, pcm_data_int16)
        #     print(f"转换回的PCM已保存到 {filename_wav}")
        # except ImportError:
        #     print(f"无法导入 scipy.io.wavfile，跳过PCM转换。")
        # except Exception as e:
        #     print(f"µ-law转PCM时出错: {e}")