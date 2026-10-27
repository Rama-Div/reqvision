import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from generators import warmup
from pipeline import analyze_text

app = FastAPI(title="ReqVision Backend")

# Development setting. Before deploying, replace "*" with your real front-end origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SRSRequest(BaseModel):
    srs_text: str


@app.on_event("startup")
def _warm_graphviz():
    import threading
    threading.Thread(target=warmup, daemon=True).start()


@app.get("/health")
def health():
    return {"status": "ok", "llm": bool(os.getenv("ANTHROPIC_API_KEY"))}


# Plain `def` (not async): FastAPI runs it in a worker thread, so the blocking
# Graphviz / network calls do not freeze the server.
@app.post("/api/analyze")
def analyze(request: SRSRequest):
    try:
        return analyze_text(request.srs_text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    # 127.0.0.1 = this computer only (also works for the Android emulator via 10.0.2.2).
    # Use HOST=0.0.0.0 only if you test from a real phone on the same Wi-Fi.
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))