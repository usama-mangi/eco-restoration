from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import uvicorn
import json
import asyncio
from eco_engine_api import EcoEngineAPI

app = FastAPI()

# Enable CORS so your React app can hit this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Engine Instance
engine = EcoEngineAPI()

@app.post("/stream")
async def stream_processing(file: UploadFile = File(...)):
    """
    Streams processing steps to the client.
    React Usage:
    const eventSource = new EventSource('/stream');
    (Requires fetch-event-source logic for POST requests)
    """
    contents = await file.read()
    
    async def event_generator():
        async for update in engine.process_stream(contents):
            # SSE requires specific format: "event: type\ndata: json\n\n"
            # sse-starlette handles this wrapping if we yield dicts
            yield {
                "event": update["event"],
                "data": json.dumps(update["data"])
            }
            # Yield control to event loop
            await asyncio.sleep(0.01)

    return EventSourceResponse(event_generator())

if __name__ == "__main__":
    print("🚀 Starting EcoRestoration API on port 3000...")
    uvicorn.run(app, host="127.0.0.1", port=3000)
