import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app
from app.routers import analyses as analyses_router
from app.services import report as report_service


@pytest.fixture()
def client(tmp_path, monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine, autoflush=False)

    def override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    monkeypatch.setattr(analyses_router, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(report_service, "UPLOAD_DIR", tmp_path)
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture()
def png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()