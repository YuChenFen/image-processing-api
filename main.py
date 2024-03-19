# import subprocess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from inference import get_images_api
import uvicorn

PORT_API = 8008

app = FastAPI(
    title="API server",
    version="1.0.0",
)

app.include_router(get_images_api.get_images_api_router, prefix="/v1/images", tags=["images"])

# Configure CORS settings
origins = [
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    # allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def start_api_server():
    try:
        print("Starting API server...")
        uvicorn.run(app, host="0.0.0.0", port=PORT_API, log_level="info")
        return True
    except:
        print("Failed to start API server")
        return False


if __name__ == "__main__":
    start_api_server()