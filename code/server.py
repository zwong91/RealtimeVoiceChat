# server.py

import logging
from logsetup import setup_logging
setup_logging(logging.INFO)
logger = logging.getLogger(__name__)
if __name__ == "__main__":
    logger.info("🖥️ Welcome to local real-time voice chat")

from datetime import datetime
import uvicorn
import asyncio
import struct
import json
import time
import sys
import os # Added for environment variable access

from typing import Any, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import HTMLResponse, Response, FileResponse

USE_SSL = False
START_ENGINE = "kokoro"
# DIRECT_STREAM = True
DIRECT_STREAM = START_ENGINE=="orpheus"

# Define the maximum allowed size for the incoming audio queue
try:
    MAX_AUDIO_QUEUE_SIZE = int(os.getenv("MAX_AUDIO_QUEUE_SIZE", 50))
    if __name__ == "__main__":
        logger.info(f"Audio queue size limit set to: {MAX_AUDIO_QUEUE_SIZE}")
except ValueError:
    if __name__ == "__main__":
        logger.warning("Invalid MAX_AUDIO_QUEUE_SIZE env var. Using default: 50")
    MAX_AUDIO_QUEUE_SIZE = 50


if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from handlerequests import LanguageProcessor
from audio_out import AudioOutProcessor
from audio_in import AudioInputProcessor
from colors import Colors

LANGUAGE = "en"
# TTS_FINAL_TIMEOUT = 0.5 # unsure if 1.0 is needed for stability
TTS_FINAL_TIMEOUT = 1.0 # unsure if 1.0 is needed for stability

# --------------------------------------------------------------------
# Custom no-cache StaticFiles
# --------------------------------------------------------------------
# Custom static files class to disable caching.
class NoCacheStaticFiles(StaticFiles):
    """
    Serves static files without allowing client-side caching.
    Useful for development to ensure the most recent version of files is always loaded.
    """
    async def get_response(self, path: str, scope: Dict[str, Any]) -> Response:
        response: Response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers.__delitem__("etag")
        response.headers.__delitem__("last-modified")
        return response

# --------------------------------------------------------------------
# Lifespan management
# --------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🖥️ Server starting up")
    app.state.LanguageProcessor = LanguageProcessor(
        is_orpheus=START_ENGINE=="orpheus",
    )
    app.state.AudioInputProcessor = AudioInputProcessor(
        LANGUAGE,
        is_orpheus=START_ENGINE=="orpheus",
    )
    app.state.AudioOutProcessor = AudioOutProcessor(
        engine=START_ENGINE,
        language=LANGUAGE
    )
    app.state.TTS_To_Client = False
    app.state.Aborting = False


    yield

    logger.info("🖥️ Server shutting down")
    app.state.AudioInputProcessor.shutdown()
    app.state.LanguageProcessor.shutdown()

# --------------------------------------------------------------------
# FastAPI app instance
# --------------------------------------------------------------------
app = FastAPI(lifespan=lifespan)

# Enable CORS if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files with no cache
app.mount("/static", NoCacheStaticFiles(directory="static"), name="static")

@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/favicon.ico")

@app.get("/")
async def get_index() -> HTMLResponse:
    """
    Return the index HTML page from the 'static' directory.
    """
    with open("static/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# --------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------
def parse_json_message(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("🖥️💥 Ignoring client message with invalid JSON")
        return {}

def format_timestamp_ns(timestamp_ns: int) -> str:
    # Split into whole seconds and the nanosecond remainder
    seconds = timestamp_ns // 1_000_000_000
    remainder_ns = timestamp_ns % 1_000_000_000

    # Convert seconds part into a datetime object (local time)
    dt = datetime.fromtimestamp(seconds)

    # Format the main time as HH:MM:SS
    time_str = dt.strftime("%H:%M:%S")

    # For instance, if you want milliseconds, divide the remainder by 1e6 and format as 3-digit
    milliseconds = remainder_ns // 1_000_000
    formatted_timestamp = f"{time_str}.{milliseconds:03d}"

    return formatted_timestamp

# --------------------------------------------------------------------
# WebSocket data processing
# --------------------------------------------------------------------

async def process_incoming_data(ws: WebSocket, app: FastAPI, incoming_chunks: asyncio.Queue) -> None:
    """
    Receive messages via WebSocket and push any audio byte chunks into the queue.
    Also log when 'tts_start' and 'tts_stop' messages arrive from the client.
    Applies back-pressure by dropping chunks if the queue exceeds MAX_AUDIO_QUEUE_SIZE.
    """
    try:
        while True:
            msg = await ws.receive()
            if "bytes" in msg and msg["bytes"]:
                raw = msg["bytes"]

                # Ensure we have at least an 8‑byte header: 4 bytes timestamp_ms + 4 bytes flags
                if len(raw) < 8:
                    logger.warning("🖥️💥 Received packet too short for 8‑byte header.")
                    continue

                # Unpack big‑endian uint32 timestamp (ms) and uint32 flags
                timestamp_ms, flags = struct.unpack("!II", raw[:8])
                client_sent_ns = timestamp_ms * 1_000_000

                # Build metadata using fixed fields
                metadata = {
                    "client_sent_ms":           timestamp_ms,
                    "client_sent":              client_sent_ns,
                    "client_sent_formatted":    format_timestamp_ns(client_sent_ns),
                    "isTTSPlaying":             bool(flags & 1),
                }

                # Record server receive time
                server_ns = time.time_ns()
                metadata["server_received"] = server_ns
                metadata["server_received_formatted"] = format_timestamp_ns(server_ns)

                # The rest of the payload is raw PCM bytes
                metadata["pcm"] = raw[8:]

                # Check queue size before putting data
                current_qsize = incoming_chunks.qsize()
                if current_qsize < MAX_AUDIO_QUEUE_SIZE:
                    # Now put only the metadata dict (containing PCM audio) into the processing queue.
                    await incoming_chunks.put(metadata)
                else:
                    # Queue is full, drop the chunk and log a warning
                    logger.warning(
                        f"Audio queue full ({current_qsize}/{MAX_AUDIO_QUEUE_SIZE}); dropping chunk. Possible lag."
                    )

            elif "text" in msg and msg["text"]:
                # Text-based message: parse JSON
                data = parse_json_message(msg["text"])
                msg_type = data.get("type")
                # logger.info(f"📥 Received from client: {data}")
                logger.info(Colors.apply(f"⬅️⬅️📥←←Client: {data}").orange)


                if msg_type == "tts_start":
                    # logger.info("🖥️ Received tts_start from client.")
                    app.state.TTS_Client_Playing = True
                elif msg_type == "tts_stop":
                    # logger.info("🖥️ Received tts_stop from client.")
                    app.state.TTS_Client_Playing = False
                # Add to the handleJSONMessage function in server.py
                elif msg_type == "clear_history":
                    # logger.info("🖥️ Received clear_history from client.")
                    app.state.LanguageProcessor.reset()
                elif msg_type == "set_speed":
                    speed_value = data.get("speed", 0)
                    speed_factor = speed_value / 100.0  # Convert 0-100 to 0.0-1.0
                    turn_detection = app.state.AudioInputProcessor.transcriber.turn_detection
                    if turn_detection:
                        turn_detection.update_settings(speed_factor)
                        logger.info(f"🖥️ Updated turn detection settings to factor: {speed_factor:.2f}")


    except asyncio.CancelledError:
        pass
    except WebSocketDisconnect as e:
        logger.warning(f"🖥️💥 {Colors.apply('WARNING').red} disconnect in process_incoming_data: {repr(e)}")
    except RuntimeError as e:  # Often raised on closed transports
        logger.error(f"🖥️ {Colors.apply('RUNTIME_ERROR').red} in process_incoming_data: {repr(e)}")
    except Exception as e:
        logger.exception(f"🖥️ {Colors.apply('EXCEPTION').red} in process_incoming_data: {repr(e)}")

async def send_text_messages(ws: WebSocket, message_queue: asyncio.Queue) -> None:
    """
    Continuously send text messages from the queue to the client via WebSocket.
    """
    try:
        while True:
            await asyncio.sleep(0.001)
            data = await message_queue.get()
            msg_type = data.get("type")
            if msg_type != "tts_chunk":
                logger.info(Colors.apply(f"➡️➡️📥→→Client: {data}").orange)
            await ws.send_json(data)
    except asyncio.CancelledError:
        pass
    except WebSocketDisconnect as e:
        logger.warning(f"🖥️💥 {Colors.apply('WARNING').red} disconnect in send_text_messages: {repr(e)}")
    except RuntimeError as e:  # Often raised on closed transports
        logger.error(f"🖥️ {Colors.apply('RUNTIME_ERROR').red} in send_text_messages: {repr(e)}")
    except Exception as e:
        logger.exception(f"🖥️ {Colors.apply('EXCEPTION').red} in send_text_messages: {repr(e)}")

async def send_tts_chunks(app: FastAPI, message_queue: asyncio.Queue, callbacks) -> None:
    """
    Continuously send TTS audio chunks from the AudioOutProcessor to the client.
    """
    try:
        logger.info("🖥️ Starting TTS chunk sender")
        last_quick_answer_chunk = 0
        last_chunk_sent = 0
        prev_status = None

        while True:
            await asyncio.sleep(0.001)

            # grab all the bits into a tuple
            curr_status = (
                int(app.state.TTS_To_Client),
                int(app.state.TTS_Client_Playing),
                int(app.state.TTS_Chunk_Sent),
                int(app.state.AudioOutProcessor.synthesis_available.is_set()),
                int(callbacks.is_hot),
                int(callbacks.synthesis_started),
            )

            # only log if something’s changed
            if curr_status != prev_status:
                status = Colors.apply("🖥️ State ").red
                logger.info(
                    f"{status} To_Client {curr_status[0]}, "
                    f"TTS_playing {curr_status[1]}, "
                    f"Chunk_Sent {curr_status[2]}, "
                    f"synth_free {curr_status[3]}, "
                    f"hot {curr_status[4]}, synth_start {curr_status[5]}"
                )
                prev_status = curr_status

            if not app.state.TTS_To_Client:
                await asyncio.sleep(0.001)
                continue

            chunk = None
            # Try quick_answer queue first
            try:
                chunk = app.state.AudioOutProcessor.quick_answer_audio_chunks.get_nowait()
                if chunk:
                    last_quick_answer_chunk = time.time()
            except asyncio.QueueEmpty:
                pass

            # If no chunk, check final_answer queue
            if chunk is None:
                try:
                    chunk = app.state.AudioOutProcessor.final_answer_audio_chunks.get_nowait()
                except asyncio.QueueEmpty:
                    pass

            # If nothing available, wait briefly
            if chunk is None:
                await asyncio.sleep(0.001)

                if app.state.TTS_Chunk_Sent:

                    last_quick_answer_chunk_decayed = (
                        last_quick_answer_chunk
                        and time.time() - last_quick_answer_chunk > TTS_FINAL_TIMEOUT
                        and time.time() - last_chunk_sent > TTS_FINAL_TIMEOUT
                    )

                    final_finished = not app.state.AudioOutProcessor.synthesis_final_running
                    maybe_finished = last_quick_answer_chunk_decayed or (DIRECT_STREAM and final_finished)

                    if (maybe_finished
                        and app.state.AudioOutProcessor.synthesis_available.is_set()
                        and not app.state.LanguageProcessor.paused_generator
                        ):

                        base64_chunk = app.state.AudioOutProcessor.flush_base64_chunk()
                        # Ensure flush actually returned something before sending
                        if base64_chunk:
                            message_queue.put_nowait({
                                "type": "tts_chunk",
                                "content": base64_chunk
                            })

                        app.state.TTS_To_Client = False
                        app.state.TTS_Chunk_Sent = False
                        callbacks.reset_state()
                        last_quick_answer_chunk = 0
                        last_chunk_sent = 0

                        logger.info(Colors.apply("🖥️ All TTS chunks sent to client").green)

                continue

            # If we have an actual chunk, convert to base64 and send
            base64_chunk = app.state.AudioOutProcessor.get_base64_chunk(chunk)
            message_queue.put_nowait({
                "type": "tts_chunk",
                "content": base64_chunk
            })
            last_chunk_sent = time.time()

            if not app.state.TTS_Chunk_Sent:
                # This is our first chunk, so we kick off final TTS or finalize logic
                async def reset_interrupt_flag_later():
                    await asyncio.sleep(1)
                    if app.state.AudioInputProcessor.interrupted:
                        logger.info(f"{Colors.apply('🖥️🎙️ ▶️ Microphone continued').cyan}")
                        app.state.AudioInputProcessor.interrupted = False
                    logger.info(Colors.apply("🖥️ interruption flag reset after TTS chunk").cyan)

                asyncio.create_task(reset_interrupt_flag_later())

                if app.state.LanguageProcessor.paused_generator:
                    logger.info(Colors.apply("🖥️ Final TTS started").green)
                    tts_generator = app.state.LanguageProcessor.get_paused_generator()
                    app.state.AudioOutProcessor.start_synthesis_final_thread(tts_generator, "continue paused generator")
                else:
                    # If there's no generator, the quick answer is already done.
                    # logger.info(f"Quick answer finished: {app.state.LanguageProcessor.last_fast_answer}")
                    print(Colors.apply("Quick answer finished").pink)
                    callbacks.send_final_assistant_answer()

            app.state.TTS_Chunk_Sent = True

    except asyncio.CancelledError:
        pass
    except WebSocketDisconnect as e:
        logger.warning(f"🖥️💥 {Colors.apply('WARNING').red} disconnect in send_tts_chunks: {repr(e)}")
    except RuntimeError as e:
        logger.error(f"🖥️ {Colors.apply('RUNTIME_ERROR').red} in send_tts_chunks: {repr(e)}")
    except Exception as e:
        logger.exception(f"🖥️ {Colors.apply('EXCEPTION').red} in send_tts_chunks: {repr(e)}")

# --------------------------------------------------------------------
# Callback class to handle transcription events
# --------------------------------------------------------------------
class TranscriptionCallbacks:
    def __init__(self, app: FastAPI, message_queue: asyncio.Queue):

        self.app = app
        self.message_queue = message_queue

        self.reset_state()

    def reset_state(self):
        self.is_hot = False
        self.synthesis_started = False
        self.assistant_answer = ""
        self.final_assistant_answer = ""
        self.is_processing_potential = False
        # self.final_transcription = ""
        self.last_inferred_transcription = ""
        # Added missing initialization from original code provided
        self.final_assistant_answer_sent = False

    def on_partial(self, txt: str):
        self.final_assistant_answer_sent = False
        self.final_transcription = ""
        self.partial_transcription = txt
        self.message_queue.put_nowait({"type": "partial_user_request", "content": txt})

    def get_final_transcription(self):
        if not self.final_transcription:
            start_time = time.time()
            # Ensure transcriber and recorder are available
            if hasattr(self.app.state.AudioInputProcessor, 'transcriber') and \
               hasattr(self.app.state.AudioInputProcessor.transcriber, 'recorder') and \
               hasattr(self.app.state.AudioInputProcessor.transcriber.recorder, 'perform_final_transcription'):

                audio = self.app.state.AudioInputProcessor.transcriber.get_last_audio_copy()
                final_transcription = self.app.state.AudioInputProcessor.transcriber.recorder.perform_final_transcription(audio)
                self.final_transcription = final_transcription
                end_time = time.time()

                if self.final_transcription:
                    logger.info(f"{Colors.apply('🖥️💭 FINAL TRANSCRIPTION: ').green}{self.final_transcription} in {end_time - start_time:.2f} seconds")
                else:
                    logger.warning(f"{Colors.apply('🖥️💥 Warning ').red} No final transcription available.")
                    if audio is not None:
                        logger.info(f"Audio length: {len(audio)} samples (float32)")
                    else:
                        logger.warning(f"{Colors.apply('🖥️💥 Warning ').red} Audio data was None for final transcription.")

                return final_transcription
            else:
                logger.error(f"{Colors.apply('🖥️💥 ERROR').red} Transcriber or recorder method not available for final transcription.")
                self.final_transcription = "" # Ensure reset
                return None

    def safe_abort_running_syntheses(self, reason: str):
        if app.state.LanguageProcessor.is_working or app.state.AudioOutProcessor.synthesis_running or app.state.AudioOutProcessor.synthesis_final_running:
            logger.info(f"{Colors.apply('🖥️ ABORT').red}, reason: {reason}")
            if app.state.LanguageProcessor.is_working:
                self.abort_generations("on potential sentence, LanguageProcessor working")
            else:
                self.abort_generations("on potential sentence, AudioOutProcessor synthesis running")

    def on_potential_sentence(self, txt: str):
        logger.info(f"{Colors.apply('🖥️💭 SENTENCE: ').yellow}{txt} {'[x]' if self.synthesis_started else '[OK]'}")

        self.safe_abort_running_syntheses("precheck on_potential_sentence")

        # Original logic: Check conditions and call get_final_transcription directly
        if not self.synthesis_started and not self.is_hot and not self.is_processing_potential:
            self.is_processing_potential = True
            transcription = self.get_final_transcription()
            # Check if transcription was successful before processing
            if not self.synthesis_started and not self.is_hot:
                if not transcription:
                    transcription = self.partial_transcription
                if transcription:
                    if DIRECT_STREAM:
                        self.safe_abort_running_syntheses("final check before new generation")
                        logger.info(f"{Colors.apply('🖥️💭 LLM STARTING FULL PIPELINE potential sentence').teal}")
                        self.last_inferred_transcription = transcription
                        tts_generator = app.state.LanguageProcessor.get_full_generator(transcription)
                        app.state.AudioOutProcessor.start_synthesis_final_thread(tts_generator, "potential sentence")
                    else:
                        self.app.state.LanguageProcessor.process_potential_sentence(self.final_transcription)
                else:
                    logger.warning(f"{Colors.apply('🖥️💥 WARNING').red} Final transcription empty/failed in on_potential_sentence, skipping LLM process.")
            self.is_processing_potential = False

    def on_potential_final(self, txt: str):
        if DIRECT_STREAM:
            return

        logger.info(f"{Colors.apply('🖥️💭 HOT: ').magenta}{txt}")
        self.is_hot = True
        transcription = self.get_final_transcription()
        # Check if transcription was successful before processing
        if not transcription:
            transcription = self.partial_transcription
        if transcription:
            self.app.state.LanguageProcessor.return_fast_sentence_answer(transcription)
        else:
            logger.warning(f"{Colors.apply('🖥️💥 WARNING').red} Final transcription empty/failed in on_potential_final, skipping LLM process.")

    def on_fast_answer(self, txt: str):
        if DIRECT_STREAM:
            return

        logger.info(f"{Colors.apply('🖥️💭 SUPERHOT: ').red.bold}{txt}")
        if self.is_hot and not self.synthesis_started:
            self.synthesis_started = True
            self.assistant_answer = txt
            # Send initial fast answer immediately
            self.message_queue.put_nowait({
                "type": "partial_assistant_answer",
                "content": self.assistant_answer
            })
            self.app.state.AudioOutProcessor.start_synthesis_quick_thread(txt)

    def on_potential_abort(self):
        logger.info(f"{Colors.apply('🖥️💭 COLD').blue}")
        self.app.state.AudioOutProcessor.abort_syntheses()
        self.app.state.LanguageProcessor.stop_event.set()
        self.reset_state()

    def on_before_final(self, audio: bytes, txt: str):
        print(Colors.apply('=========================================================================').light_gray)
        # first block further incoming audio
        if not self.app.state.AudioInputProcessor.interrupted:
            logger.info(f"{Colors.apply('🖥️🎙️ ⏸️ Microphone interrupted').cyan}")
            self.app.state.AudioInputProcessor.interrupted = True

        # Ensure final transcription is available
        transcription = self.get_final_transcription()
        if not transcription:
            logger.error(f"{Colors.apply('🖥️💥 ERROR').red} Final transcription unavailable in on_before_final. Aborting TTS trigger.")
            transcription = self.last_inferred_transcription

        if DIRECT_STREAM:
            if not app.state.LanguageProcessor.is_working and not app.state.AudioOutProcessor.synthesis_running and app.state.AudioOutProcessor.are_queues_empty():
                logger.warning(f"{Colors.apply('🖥️💥 WARNING').red} We need to START RETRIEVAL TOO LATE. This should NOT OCCUR!")
                if not transcription:
                    transcription = self.partial_transcription
                if transcription:
                    logger.info(f"{Colors.apply('🖥️💭 LLM STARTING FULL PIPELINE on before final').teal}")
                    tts_generator = app.state.LanguageProcessor.get_full_generator(transcription)
                    app.state.AudioOutProcessor.start_synthesis_final_thread(tts_generator, "before final user transcription")
                else:
                    logger.warning(f"{Colors.apply('🖥️💥 WARNING').red} Final transcription empty/failed in on_before_final, skipping LLM process.")
                    return
        else:
            # Original logic: Generate fast answer if conditions met
            if not self.is_hot and not self.synthesis_started:
                self.is_hot = True
                # Use final_transcription for fast answer generation
                fast = self.app.state.LanguageProcessor.return_fast_answer(self.final_transcription)
                if fast: # Only proceed if fast answer generated
                    self.synthesis_started = True
                    self.assistant_answer = fast
                    self.app.state.AudioOutProcessor.start_synthesis_quick_thread(fast)
                else:
                    logger.warning(f"{Colors.apply('🖥️💥 WARNING').yellow} Fast answer generation failed in on_before_final.")

        logger.info(f"{Colors.apply('🖥️💭 TTS STREAM RELEASED').blue}")
        self.app.state.TTS_To_Client = True

        # Send final user request (using the reliable final_transcription)
        self.message_queue.put_nowait({
            "type": "final_user_request",
            "content": self.final_transcription
        })
        # Add user message to history if not already the last message
        if not self.app.state.LanguageProcessor.history or self.app.state.LanguageProcessor.history[-1].get('content') != self.final_transcription:
            self.app.state.LanguageProcessor.history.append({"role": "user", "content": self.final_transcription})

        # Send current assistant answer (might be the fast one or empty)
        if self.final_assistant_answer and not self.final_assistant_answer_sent:
            print(Colors.apply("send_final_assistant_answer in on_before_final").pink)
            self.send_final_assistant_answer()
        elif self.assistant_answer:
             self.message_queue.put_nowait({
                 "type": "partial_assistant_answer",
                 "content": self.assistant_answer
             })

    def on_final(self, txt: str):
        # Original logic: Log the final transcription from the STT engine's callback
        logger.info(f"\n{Colors.apply('🖥️💭 FINAL USER ANSWER (STT Callback): ').green}{txt}")
        # Optionally update self.final_transcription if needed, though get_final_transcription is preferred
        if not self.final_transcription:
             self.final_transcription = txt


    def final_answer_token(self, token: str):
        self.assistant_answer += token
        if not DIRECT_STREAM or self.app.state.TTS_To_Client:
            self.message_queue.put_nowait({
                "type": "partial_assistant_answer",
                "content": self.assistant_answer # Send cumulative answer
            })

    def abort_generations(self, reason):
        logger.info(f"{Colors.apply('🖥️💭 abort_generations called').blue}")
        if not self.app.state.Aborting:
            if not self.app.state.TTS_To_Client and not self.app.state.TTS_Chunk_Sent and not self.app.state.AudioOutProcessor.synthesis_running and not app.state.LanguageProcessor.is_working and self.app.state.AudioOutProcessor.are_queues_empty():
                logger.info(f"{Colors.apply('🖥️💭 Aborting TTS syntheses').blue} + {Colors.apply('SHOULD NOT BE NEEDED, ALL LOOKS FINE').pink}, reason: {reason}")
            else:
                logger.info(f"{Colors.apply('🖥️💭 Aborting TTS syntheses').blue}, reason: {reason}")

            self.app.state.Aborting = True
            self.app.state.TTS_To_Client = False
            self.app.state.TTS_Chunk_Sent = False
            self.app.state.AudioOutProcessor.abort_syntheses()
            self.app.state.LanguageProcessor.stop_event.set() # Signal LLM to stop
            self.message_queue.put_nowait({ # Tell client to stop playback and clear buffer
                "type": "tts_interruption",
                "content": ""
            })
            # Reset state *after* signaling interruption
            self.reset_state()
            self.app.state.Aborting = False

    def on_recording_start(self):
        logger.info(f"{Colors.ORANGE}🎙️ Recording started.{Colors.RESET} TTS Client Playing: {self.app.state.TTS_Client_Playing}")
        if self.app.state.TTS_Client_Playing:
            logger.info(f"{Colors.apply('🖥️💭 INTERRUPTING TTS due to recording start').blue}")
            # 1. Tell client to stop playing immediately and ignore further chunks for a bit
            self.message_queue.put_nowait({
                "type": "stop_tts", # Client handles this to mute/ignore
                "content": ""
            })

            # 2. Send final assistant answer *if* one was generated and not sent
            print(Colors.apply("send_final_assistant_answer in on_recording_start").pink)
            self.send_final_assistant_answer()

            # 3. Stop server-side TTS generation and reset related state
            logger.info(f"{Colors.apply('🖥️💭 RECORDING START ABORTING').red}")
            self.abort_generations("on_recording_start, TTS Playing") # This stops synth, resets flags, signals LLM


    def send_final_assistant_answer(self):
        if not self.final_assistant_answer_sent and self.final_assistant_answer:
            import re
            # Clean up the final answer text
            cleaned_answer = re.sub(r'[\r\n]+', ' ', self.final_assistant_answer)
            cleaned_answer = re.sub(r'\s+', ' ', cleaned_answer).strip()
            cleaned_answer = cleaned_answer.replace('\\n', ' ')
            cleaned_answer = re.sub(r'\s+', ' ', cleaned_answer).strip()

            if cleaned_answer: # Ensure it's not empty after cleaning
                logger.info(f"\n{Colors.apply('🖥️💭 FINAL ASSISTANT ANSWER: ').green}{cleaned_answer}")
                self.message_queue.put_nowait({
                    "type": "final_assistant_answer",
                    "content": cleaned_answer
                })
                self.final_assistant_answer_sent = True
                # Add to history only if it's different from the last entry
                if not self.app.state.LanguageProcessor.history or self.app.state.LanguageProcessor.history[-1].get('content') != cleaned_answer:
                    self.app.state.LanguageProcessor.history.append({"role": "assistant", "content": cleaned_answer})
            else:
                logger.warning(f"{Colors.YELLOW}Final assistant answer was empty after cleaning.{Colors.RESET}")
                # Don't send empty final answer, don't mark as sent
                self.final_assistant_answer_sent = False


    def last_final_answer_token_sent(self):
         logger.info(f"{Colors.BLUE}Last final answer token processing finished.{Colors.RESET}")
         self.final_assistant_answer = self.assistant_answer
         if self.app.state.TTS_To_Client:
             self.send_final_assistant_answer()

# --------------------------------------------------------------------
# Main WebSocket endpoint
# --------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("🖥️ Client connected via WebSocket.")

    app.state.TTS_Chunk_Sent = False
    app.state.TTS_Client_Playing = False

    message_queue = asyncio.Queue()
    audio_chunks = asyncio.Queue()

    # Set up callback manager
    callbacks = TranscriptionCallbacks(app, message_queue)

    # Attach the callbacks to the AudioInputProcessor
    app.state.AudioInputProcessor.realtime_callback = callbacks.on_partial
    app.state.AudioInputProcessor.transcriber.potential_sentence_end = callbacks.on_potential_sentence
    app.state.AudioInputProcessor.transcriber.potential_full_transcription_callback = callbacks.on_potential_final
    app.state.AudioInputProcessor.transcriber.potential_full_transcription_abort_callback = callbacks.on_potential_abort
    # Attach the original on_final callback from the transcriber
    app.state.AudioInputProcessor.transcriber.full_transcription_callback = callbacks.on_final
    app.state.AudioInputProcessor.transcriber.before_final_sentence = callbacks.on_before_final
    app.state.AudioInputProcessor.recording_start_callback = callbacks.on_recording_start

    # Attach callback for fast answers
    app.state.LanguageProcessor.fast_answer_callback = callbacks.on_fast_answer
    app.state.LanguageProcessor.final_answer_token = callbacks.final_answer_token
    app.state.LanguageProcessor.last_final_answer_token_sent = callbacks.last_final_answer_token_sent

    # Create tasks for handling different responsibilities
    tasks = [
        asyncio.create_task(process_incoming_data(ws, app, audio_chunks)),
        asyncio.create_task(app.state.AudioInputProcessor.process_chunk_queue(audio_chunks)),
        asyncio.create_task(send_text_messages(ws, message_queue)),
        asyncio.create_task(send_tts_chunks(app, message_queue, callbacks)),
    ]

    try:
        # Wait for any task to complete (e.g., client disconnect)
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            if not task.done():
                task.cancel()
        # Await cancelled tasks to let them clean up if needed
        await asyncio.gather(*pending, return_exceptions=True)
    except Exception as e:
        logger.error(f"🖥️ {Colors.apply('ERROR').red} in WebSocket session: {repr(e)}")
    finally:
        logger.info("🖥️ Cleaning up WebSocket tasks...")
        for task in tasks:
            if not task.done():
                task.cancel()
        # Ensure all tasks are awaited after cancellation
        # Use return_exceptions=True to prevent gather from stopping on first error during cleanup
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("🖥️ WebSocket session ended.")

# --------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------
if __name__ == "__main__":

    # Run the server without SSL
    if not USE_SSL:
        # Run the server without SSL
        uvicorn.run("server:app", host="0.0.0.0", port=8000, log_config=None)

    else:
        # Check if cert files exist
        cert_file = "127.0.0.1+1.pem"
        key_file = "127.0.0.1+1-key.pem"
        if not os.path.exists(cert_file) or not os.path.exists(key_file):
             logger.error(f"SSL cert file ({cert_file}) or key file ({key_file}) not found.")
             logger.error("Please generate them using mkcert:")
             logger.error("  choco install mkcert")
             logger.error("  mkcert -install")
             logger.error("  mkcert 127.0.0.1 YOUR_LOCAL_IP") # Remind user to replace with actual IP if needed
             logger.error("Exiting.")
             sys.exit(1)

        # Run the server with SSL
        uvicorn.run(
            "server:app",
            host="0.0.0.0",
            port=8000,
            log_config=None,
            ssl_certfile=cert_file,
            ssl_keyfile=key_file,
        )