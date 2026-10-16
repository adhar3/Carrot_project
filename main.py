from fastapi import FastAPI
from pydantic_experiments import Telemetry

app = FastAPI(title="Telemetry validation demo")


@app.post("/telemetry", status_code=201)
def receive_telemetry(payload: Telemetry):
    return {
        "status": "accepted",
        "image_id": payload.image_id,
        "detections": len(payload.detections),
    }