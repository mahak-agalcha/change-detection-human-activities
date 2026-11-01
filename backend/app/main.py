from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import engine, Base
from . import models
from .routers import analyses
from fastapi.staticfiles import StaticFiles
from .config import UPLOAD_DIR
from .config import UPLOAD_DIR, settings

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Change Detection API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.CORS_ORIGINS.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyses.router)
app.mount("/files", StaticFiles(directory=UPLOAD_DIR), name="files")


@app.get("/health")
def health():
    return {"status": "ok"}