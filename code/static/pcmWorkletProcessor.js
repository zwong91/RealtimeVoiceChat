// static/pcmWorkletProcessor.js
class PCMWorkletProcessor extends AudioWorkletProcessor {
    process(inputs, outputs, parameters) {
      // inputs[0] is the array of channels for the single input node
      // We only care about the first channel (mono)
      const inputChannelData = inputs[0][0];
      if (inputChannelData) {
        // Post the Float32Array back to the main thread for conversion/sending
        this.port.postMessage(inputChannelData);
      }
      // Return true to keep the processor alive
      return true;
    }
  }
  
  // Register the processor under a chosen name
  registerProcessor('pcm-worklet-processor', PCMWorkletProcessor);
  