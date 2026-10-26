from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import engine, Base
from . import models
from .routers import analyses

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Change Detection API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyses.router)


@app.get("/health")
def health():
    return {"status": "ok"}