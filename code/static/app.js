// Add timestamps to all console.log messages in HH:MM:SS.nnn format
(function() {
  const originalLog = console.log.bind(console);
  console.log = (...args) => {
    const now = new Date();
    const hh = String(now.getHours()).padStart(2, '0');
    const mm = String(now.getMinutes()).padStart(2, '0');
    const ss = String(now.getSeconds()).padStart(2, '0');
    const ms = String(now.getMilliseconds()).padStart(3, '0');
    originalLog(`[${hh}:${mm}:${ss}.${ms}]`, ...args);
  };
})();


// app.js - Complete chat version with audio + chat bubbles + typing indicators
const debug_chunk_logging = false;

const statusDiv = document.getElementById("status");
const messagesDiv = document.getElementById("messages");

let socket = null;
let audioContext = null;
let mediaStream = null;
let micWorkletNode = null;
let ttsWorkletNode = null;

let isTTSPlaying = false;
let ignoreIncomingTTS = false;

let chatHistory = []; // [{role:"user"/"assistant", content:"...", type:"final"}]
let typingUser = "";
let typingAssistant = "";

// ------ Audio + socket helpers ------

function initAudioContext() {
  if (!audioContext) {
    audioContext = new AudioContext();
    // console.log(`AudioContext sample rate: ${audioContext.sampleRate} Hz`);
  }
}

function float32ToInt16(float32Array) {
  const len = float32Array.length;
  const result = new Int16Array(len);
  for (let i = 0; i < len; i++) {
    let s = Math.max(-1, Math.min(1, float32Array[i]));
    result[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }
  return result.buffer;
}

function base64ToInt16Array(b64) {
  const raw = atob(b64);
  const buf = new ArrayBuffer(raw.length);
  const view = new Uint8Array(buf);
  for (let i = 0; i < raw.length; i++) {
    view[i] = raw.charCodeAt(i);
  }
  return new Int16Array(buf);
}

async function startRawPcmCapture() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: { ideal: 24000 },
        channelCount: 1,
        echoCancellation: true,
        autoGainControl: true,
        noiseSuppression: true
      }
    });
    mediaStream = stream;
    initAudioContext();
    await audioContext.audioWorklet.addModule('/static/pcmWorkletProcessor.js');
    micWorkletNode = new AudioWorkletNode(audioContext, 'pcm-worklet-processor');
    micWorkletNode.port.onmessage = (event) => {
      const inputData = event.data;
      if (socket && socket.readyState === WebSocket.OPEN) {
        const pcm16Buffer = float32ToInt16(inputData);
        sendAudioChunkWithMetadata(pcm16Buffer);
      } 
    };
    const source = audioContext.createMediaStreamSource(stream);
    source.connect(micWorkletNode);
    statusDiv.textContent = "Recording...";
  } catch (err) {
    statusDiv.textContent = "Mic access denied.";
    console.error(err);
  }
}

async function setupTTSPlayback() {
  await audioContext.audioWorklet.addModule('/static/ttsPlaybackProcessor.js');
  ttsWorkletNode = new AudioWorkletNode(audioContext, 'tts-playback-processor');

  ttsWorkletNode.port.onmessage = (event) => {
    const { type } = event.data;
    if (type === 'ttsPlaybackStarted') {
      if (!isTTSPlaying && socket && socket.readyState === WebSocket.OPEN) {
        isTTSPlaying = true;
        console.log("TTS playback started. Reason: ttsWorkletNode Event ttsPlaybackStarted.");
        socket.send(JSON.stringify({ type: 'tts_start' }));
      }
    } else if (type === 'ttsPlaybackStopped') {
      if (isTTSPlaying && socket && socket.readyState === WebSocket.OPEN) {
        isTTSPlaying = false;
        console.log("TTS playback stopped. Reason: ttsWorkletNode Event ttsPlaybackStopped.");
        socket.send(JSON.stringify({ type: 'tts_stop' }));
      }
    }
  };
  ttsWorkletNode.connect(audioContext.destination);
}

function cleanupAudio() {
  if (micWorkletNode) {
    micWorkletNode.disconnect();
    micWorkletNode = null;
  }
  if (ttsWorkletNode) {
    ttsWorkletNode.disconnect();
    ttsWorkletNode = null;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }
  if (mediaStream) {
    mediaStream.getAudioTracks().forEach(track => track.stop());
    mediaStream = null;
  }
}

// ----- Chat bubble rendering -----

function renderMessages() {
  messagesDiv.innerHTML = "";
  chatHistory.forEach(msg => {
    const bubble = document.createElement("div");
    bubble.className = `bubble ${msg.role}`;
    bubble.textContent = msg.content;
    messagesDiv.appendChild(bubble);
  });
  // Typing indicator for user (composing)
  if (typingUser) {
    const typing = document.createElement("div");
    typing.className = "bubble user typing";
    typing.innerHTML = typingUser + '<span style="opacity:.6;">✏️</span>';
    messagesDiv.appendChild(typing);
  }
  // Typing indicator for assistant (streaming)
  if (typingAssistant) {
    const typing = document.createElement("div");
    typing.className = "bubble assistant typing";
    typing.innerHTML = typingAssistant + '<span style="opacity:.6;">✏️</span>';
    messagesDiv.appendChild(typing);
  }
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function sendAudioChunkWithMetadata(pcmDataBuffer) {
  // Check for empty PCM data.
  if (!pcmDataBuffer || pcmDataBuffer.byteLength === 0) {
    console.warn("sendAudioChunkWithMetadata: Empty PCM data, not sending.");
    return;
  }

  // Create a high-resolution timestamp (in nanoseconds).
  const nowMs = Date.now();  // Milliseconds since epoch.
  const nowFractional = performance.now() % 1; // Fractional part in ms.
  const timestampNanosec = BigInt(nowMs) * 1000000n + BigInt(Math.floor(nowFractional * 1000000));

  // Build the metadata object.
  const metadata = {
    client_sent: timestampNanosec.toString(),  // As a string, to avoid precision loss.
    isTTSPlaying: isTTSPlaying              // Assume this variable is defined in your scope.
  };

  // Convert metadata to JSON string and encode it as UTF-8.
  const metadataJSON = JSON.stringify(metadata);
  const encoder = new TextEncoder();
  const metaBytes = encoder.encode(metadataJSON);


  // Create a 4-byte header for the metadata length using DataView in big-endian order.
  const headerBuffer = new ArrayBuffer(4);
  const headerView = new DataView(headerBuffer);
  headerView.setUint32(0, metaBytes.length, false); // false means big-endian.
  const header = new Uint8Array(headerBuffer);

  // Calculate total length: 4 bytes for header + metadata bytes + PCM data.
  const totalLength = header.byteLength + metaBytes.length + pcmDataBuffer.byteLength;

  // Allocate a new Uint8Array for the full payload.
  const combinedBuffer = new Uint8Array(totalLength);
  combinedBuffer.set(header, 0);                         // Insert header.
  combinedBuffer.set(metaBytes, header.byteLength);      // Insert metadata.
  combinedBuffer.set(new Uint8Array(pcmDataBuffer), header.byteLength + metaBytes.length); // Insert PCM data.

  socket.send(combinedBuffer.buffer);

  if (debug_chunk_logging) {
    console.log("sendAudioChunkWithMetadata: Metadata =", metadata);

    // Log the metadata byte length and PCM data length.
    console.log("sendAudioChunkWithMetadata: metaBytes.length =", metaBytes.length);
    console.log("sendAudioChunkWithMetadata: pcmDataBuffer.byteLength =", pcmDataBuffer.byteLength);

    console.log("sendAudioChunkWithMetadata: totalLength =", totalLength);
    console.log("sendAudioChunkWithMetadata: Sending combined buffer of length", combinedBuffer.byteLength);
  }
}


// function sendAudioChunkWithMetadata(pcmDataBuffer) {
//   // Ensure PCM data is valid and non-empty.
//   if (!pcmDataBuffer || pcmDataBuffer.byteLength === 0) {
//     console.warn("sendAudioChunkWithMetadata: Empty PCM data, not sending.");
//     return;
//   }

//   // Generate a high-resolution timestamp.
//   const nowMs = Date.now();  // Milliseconds since epoch.
//   const nowFractional = performance.now() % 1; // Fractional milliseconds.
//   // Convert the time to nanoseconds.
//   const timestampNanosec = BigInt(nowMs) * 1000000n + BigInt(Math.floor(nowFractional * 1000000));

//   // Build a metadata object with the timestamp and additional state info.
//   const metadata = {
//     timestamp: timestampNanosec.toString(),  // Using a string since the number might be very large.
//     isTTSPlaying: isTTSPlaying  // isTTSPlaying should be defined in your scope.
//   };

//   // Log the metadata for debugging.
//   console.log("sendAudioChunkWithMetadata: Metadata =", metadata);

//   // Convert the metadata to a JSON string.
//   const metadataJSON = JSON.stringify(metadata);
//   const encoder = new TextEncoder();
//   const metaBytes = encoder.encode(metadataJSON);

//   // Extended logging of lengths.
//   console.log("sendAudioChunkWithMetadata: metaBytes.length =", metaBytes.length);
//   console.log("sendAudioChunkWithMetadata: pcmDataBuffer.byteLength =", pcmDataBuffer.byteLength);

//   // Prepare a 4-byte header containing the length of metaBytes, using big-endian order.
//   const metaLengthBuffer = new Uint32Array([metaBytes.length]);
//   const header = new Uint8Array(metaLengthBuffer.buffer);
  
//   // Calculate total message length.
//   const totalLength = header.byteLength + metaBytes.length + pcmDataBuffer.byteLength;
//   console.log("sendAudioChunkWithMetadata: totalLength =", totalLength);

//   // Allocate a new buffer and combine header, metadata, and PCM data.
//   const combinedBuffer = new Uint8Array(totalLength);
//   combinedBuffer.set(header, 0);
//   combinedBuffer.set(metaBytes, header.byteLength);
//   combinedBuffer.set(new Uint8Array(pcmDataBuffer), header.byteLength + metaBytes.length);

//   console.log("sendAudioChunkWithMetadata: Sending combined buffer of length", combinedBuffer.byteLength);
//   socket.send(combinedBuffer.buffer);
// }



// ----- Message event handling -----

function handleJSONMessage({ type, content }) {
  // User is composing a message
  if (type === "partial_user_request") {
    if (content && content.trim()) {
      typingUser = escapeHtml(content);
    } else {
      typingUser = "";
    }
    renderMessages();
    return;
  }
  // User completed their message
  if (type === "final_user_request") {
    if (content && content.trim()) {
      chatHistory.push({ role: "user", content: content, type: "final" });
    }
    typingUser = "";
    renderMessages();
    return;
  }
  // Assistant is streaming a response
  if (type === "partial_assistant_answer") {
    if (content && content.trim()) {
      typingAssistant = escapeHtml(content);
    } else {
      typingAssistant = "";
    }
    renderMessages();
    return;
  }
  // Assistant response complete
  if (type === "final_assistant_answer") {
    if (content && content.trim()) {
      chatHistory.push({ role: "assistant", content: content, type: "final" });
    }
    typingAssistant = "";
    renderMessages();
    return;
  }
  // TTS audio chunk
  if (type === "tts_chunk") {
    if (ignoreIncomingTTS) {
      return;
    }    
    const int16Data = base64ToInt16Array(content);
    if (ttsWorkletNode) {
      ttsWorkletNode.port.postMessage(int16Data);
    }
    return;
  }
  if (type === "tts_interruption") {
    if (ttsWorkletNode) {
      ttsWorkletNode.port.postMessage({ type: "clear" });
    }
    isTTSPlaying = false;
    console.log("TTS playback stopped. Reason: tts_interruption.");
    socket.send(JSON.stringify({ type: 'tts_stop' }));
    ignoreIncomingTTS = false;
    return;
  }
  if (type === "stop_tts") {
    if (ttsWorkletNode) {
      ttsWorkletNode.port.postMessage({ type: "clear" });
    }
    isTTSPlaying = false;
    ignoreIncomingTTS = true;
    return;
  }
}

// ----- HTML escaping (security) -----
function escapeHtml(str) {
  return (str ?? '')
    .replace(/&/g, "&amp;")
    .replace(/</g, "<")
    .replace(/>/g, ">")
    .replace(/"/g, "&quot;");
}

// ----------- UI Button Logic & Socket Management ------------
document.getElementById("clearBtn").onclick = () => {
  // Clear client-side history
  chatHistory = [];
  typingUser = "";
  typingAssistant = "";
  renderMessages();

  // Send clear command to server
  if (socket && socket.readyState === WebSocket.OPEN) { 
    socket.send(JSON.stringify({ type: 'clear_history' }));
  }
};

document.getElementById("startBtn").onclick = async () => {
  // Prevent double-start
  if (socket && socket.readyState === WebSocket.OPEN) {
    statusDiv.textContent = "Already recording.";
    return;
  }
  statusDiv.textContent = "Initializing connection...";
  // Clear previous chat only when you want a fresh conversation:
  // typingUser = typingAssistant = "";
  // renderMessages();

  wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  socket = new WebSocket(`${wsProto}//${location.host}/ws`);

  socket.onopen = async () => {
    statusDiv.textContent = "Connected. Activating mic and TTS…";
    await startRawPcmCapture();
    await setupTTSPlayback();
  };
  socket.onmessage = (evt) => {
    if (typeof evt.data === "string") {
      try {
        const msg = JSON.parse(evt.data);
        handleJSONMessage(msg);
      } catch (e) {
        console.error("Error parsing message:", e);
      }
    }
  };
  socket.onclose = () => {
    statusDiv.textContent = "Connection closed.";
    cleanupAudio();
    // Optionally: show disconnected in UI, lock buttons, etc
  };
  socket.onerror = (err) => {
    statusDiv.textContent = "Connection error.";
    cleanupAudio();
    console.error(err);
  };
};

document.getElementById("stopBtn").onclick = () => {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.close();
  }
  cleanupAudio();
  statusDiv.textContent = "Stopped.";
};

// First render
renderMessages();