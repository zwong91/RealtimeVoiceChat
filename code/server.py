
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

from typing import Any, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import HTMLResponse, Response, FileResponse

USE_SSL = False
START_ENGINE = "coqui"

if sys.platform == "win32":
    # Use the selector loop instead of proactor on Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from handlerequests import LanguageProcessor
from audio_out import AudioOutProcessor
from audio_in import AudioInputProcessor
from colors import Colors

LANGUAGE = "en"
TTS_FINAL_TIMEOUT = 0.5 # unsure if 1.0 is needed for stability

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
    app.state.AudioInputProcessor = AudioInputProcessor(LANGUAGE)
    app.state.AudioOutProcessor = AudioOutProcessor(
        engine=START_ENGINE,
        language=LANGUAGE
    )
    app.state.TTS_To_Client = False
    
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
    """
    try:
        while True:
            await asyncio.sleep(0.001)
            msg = await ws.receive()
            if "bytes" in msg and msg["bytes"]:
                raw = msg["bytes"]

                # First, check that we have at least 4 bytes for the header.
                if len(raw) < 4:
                    logger.warning("🖥️💥 Received data too short for metadata header.")
                    continue

                # Read the first 4 bytes for the metadata length.
                meta_length = struct.unpack("!I", raw[:4])[0]

                # Make sure the payload has enough bytes for the metadata.
                if len(raw) < 4 + meta_length:
                    logger.warning(
                        f"🖥️💥 Incomplete metadata received, length of packet: {len(raw)}, expected min: 4 + meta data len {meta_length}."
                    )
                    continue

                # Extract and decode the metadata.
                meta_json_bytes = raw[4:4 + meta_length]
                try:
                    metadata = json.loads(meta_json_bytes.decode('utf-8'))
                    if "client_sent" in metadata:

                        server_received_ns = time.time_ns()
                        metadata["server_received"] = server_received_ns
                        metadata["server_received_formatted"] = format_timestamp_ns(server_received_ns)
                        # Convert timestamp from nanoseconds to a formatted string.
                        timestamp_ns = int(metadata["client_sent"])
                        formatted_time = format_timestamp_ns(timestamp_ns)
                        metadata["client_sent_formatted"] = formatted_time
                except Exception as e:
                    logger.error(f"Error decoding metadata: {e}")
                    metadata = {}

                # The remainder of the payload is the PCM audio data.
                pcm_audio = raw[4 + meta_length:]
                metadata["pcm"] = pcm_audio

                # Now put only the PCM audio into the processing queue.
                await incoming_chunks.put(metadata)

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
                    if (last_quick_answer_chunk 
                        and time.time() - last_quick_answer_chunk > TTS_FINAL_TIMEOUT
                        and app.state.AudioOutProcessor.synthesis_available.is_set()
                        and not app.state.LanguageProcessor.paused_generator
                        ):

                        base64_chunk = app.state.AudioOutProcessor.flush_base64_chunk()
                        message_queue.put_nowait({
                            "type": "tts_chunk",
                            "content": base64_chunk
                        })

                        app.state.TTS_To_Client = False
                        app.state.TTS_Chunk_Sent = False
                        callbacks.reset_state()
                        last_quick_answer_chunk = 0

                        logger.info(Colors.apply("🖥️ All TTS chunks sent to client").green)
                continue

            # If we have an actual chunk, convert to base64 and send
            base64_chunk = app.state.AudioOutProcessor.get_base64_chunk(chunk)
            message_queue.put_nowait({
                "type": "tts_chunk",
                "content": base64_chunk
            })

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
                    app.state.AudioOutProcessor.start_synthesis_final_thread(tts_generator)
                else:
                    # If there's no generator, the quick answer is already done.
                    logger.info(f"Quick answer finished: {app.state.LanguageProcessor.last_fast_answer}")
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
        self.is_processing_potential = False
        self.final_transcription = ""

    def on_partial(self, txt: str):
        self.final_assistant_answer_sent = False
        self.final_transcription = ""
        self.message_queue.put_nowait({"type": "partial_user_request", "content": txt})

    def get_final_transcription(self):
        if not self.final_transcription:
            start_time = time.time()
            audio = self.app.state.AudioInputProcessor.transcriber.get_audio_copy()
            self.final_transcription = self.app.state.AudioInputProcessor.transcriber.recorder.perform_final_transcription(audio)
            end_time = time.time()
            if self.final_transcription:
                logger.info(f"{Colors.apply('🖥️💭 FINAL TRANSCRIPTION: ').green}{self.final_transcription} in {end_time - start_time:.2f} seconds")
                #logger.info(f"{Colors.apply('🖥️💭 FINAL TRANSCRIPTION: ').green}{self.final_transcription} (partial: {txt}) in {end_time - start_time:.2f} seconds")
                self.app.state.LanguageProcessor.process_potential_sentence(self.final_transcription)
            else:
                logger.warning(f"{Colors.apply('🖥️💥 WARNING #### !!!!! ').red} No final transcription available.")
                logger.info(f"Audio length: {len(audio)} bytes")


    def on_potential_sentence(self, txt: str):
        logger.info(f"{Colors.apply('🖥️💭 SENTENCE: ').yellow}{txt} {'[x]' if self.synthesis_started else '[OK]'}")

        if not self.synthesis_started and not self.is_hot and not self.is_processing_potential:
            self.is_processing_potential = True
            self.get_final_transcription()
            self.is_processing_potential = False

    def on_potential_final(self, txt: str):
        logger.info(f"{Colors.apply('🖥️💭 HOT: ').magenta}{txt}")
        self.is_hot = True
        self.get_final_transcription()
        self.app.state.LanguageProcessor.return_fast_sentence_answer(self.final_transcription)

    def on_fast_answer(self, txt: str):
        logger.info(f"{Colors.apply('🖥️💭 SUPERHOT: ').red.bold}{txt}")
        if self.is_hot and not self.synthesis_started:
            self.synthesis_started = True
            self.assistant_answer = txt
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

        # this method is not allowed to block
        if not self.is_hot and not self.synthesis_started:
            self.is_hot = True
            fast = self.app.state.LanguageProcessor.return_fast_answer(txt)
            self.synthesis_started = True
            self.assistant_answer = fast
            self.app.state.AudioOutProcessor.start_synthesis_quick_thread(fast)

        logger.info(f"{Colors.apply('🖥️💭 TTS STREAM RELEASED').blue}")
        self.app.state.TTS_To_Client = True

        self.message_queue.put_nowait({
            "type": "final_user_request",
            "content": self.final_transcription
        })
        self.app.state.LanguageProcessor.history.append({"role": "user", "content": self.final_transcription})

        self.message_queue.put_nowait({
            "type": "partial_assistant_answer",
            "content": self.assistant_answer
        })

    def on_final(self, txt: str):
        logger.info(f"\n{Colors.apply('🖥️💭 FINAL USER ANSWER: ').green}{txt}")

    def final_answer_token(self, token: str):
        self.assistant_answer += token
        self.message_queue.put_nowait({
            "type": "partial_assistant_answer",
            "content": self.assistant_answer
        })
    
    def stop_tts(self):
        self.reset_state()
        self.app.state.TTS_To_Client = False
        self.app.state.TTS_Chunk_Sent = False            
        self.app.state.AudioOutProcessor.abort_syntheses()
        self.app.state.LanguageProcessor.stop_event.set()
        self.message_queue.put_nowait({
            "type": "tts_interruption",
            "content": ""
        })

    def on_recording_start(self):
        if app.state.TTS_Client_Playing:
            self.message_queue.put_nowait({
                "type": "stop_tts",
                "content": ""
            })

            self.send_final_assistant_answer()

            logger.info(f"{Colors.apply('🖥️💭 INTERRUPTION').blue}")
            self.stop_tts()

    def send_final_assistant_answer(self):
        if not self.final_assistant_answer_sent and self.assistant_answer:
            import re
            self.assistant_answer = re.sub(r'[\r\n]+', ' ', self.assistant_answer)
            self.assistant_answer = re.sub(r'\s+', ' ', self.assistant_answer)
            self.assistant_answer = self.assistant_answer.replace('\\n', ' ')

            logger.info(f"\n{Colors.apply('🖥️💭 FINAL ASSISTANT ANSWER: ').green}{self.assistant_answer}")
            self.message_queue.put_nowait({
                "type": "final_assistant_answer",
                "content": self.assistant_answer
            })
            self.final_assistant_answer_sent = True
            self.app.state.LanguageProcessor.history.append({"role": "assistant", "content": self.assistant_answer})


    def last_final_answer_token_sent(self):
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
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    except Exception as e:
        logger.error(f"🖥️ {Colors.apply('ERROR').red} in WebSocket session: {repr(e)}")
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
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
        # Run the server with SSL (if needed)
        # run cmd as admin:
        # - choco install mkcert
        # - mkcert -install
        # - mkcert 127.0.0.1 192.168.178.123 (your local IP address)
        uvicorn.run(
            "server:app",
            host="0.0.0.0",
            port=8000,
            log_config=None,
            ssl_certfile="127.0.0.1+1.pem",
            ssl_keyfile="127.0.0.1+1-key.pem",
        )
