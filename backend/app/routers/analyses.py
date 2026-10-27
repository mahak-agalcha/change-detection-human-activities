import io
import json
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import Image
from sqlalchemy.orm import Session

from ..config import ALLOWED_EXT, MAX_FILE_MB, UPLOAD_DIR
from ..database import get_db
from ..models import Analysis, ChangeRegion
from ..services.inference import run_inference

router = APIRouter(prefix="/analyses", tags=["analyses"])


async def save_image(file: UploadFile) -> str:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, "File type not allowed")

    limit = MAX_FILE_MB * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"File larger than {MAX_FILE_MB} MB")

    try:
        Image.open(io.BytesIO(data)).verify()
    except Exception:
        raise HTTPException(400, "File is not a valid image")

    name = f"{uuid.uuid4().hex}{ext}"
    (UPLOAD_DIR / name).write_bytes(data)
    return name


@router.post("", status_code=201)
async def create_analysis(
    old_image: UploadFile = File(...),
    new_image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    old_name = await save_image(old_image)
    new_name = await save_image(new_image)
    analysis = Analysis(old_image_path=old_name, new_image_path=new_name)
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    return {"id": analysis.id, "status": analysis.status}


@router.post("/{analysis_id}/run")
def run_analysis(analysis_id: int, db: Session = Depends(get_db)):
    analysis = db.get(Analysis, analysis_id)
    if not analysis:
        raise HTTPException(404, "Analysis not found")

    results = run_inference(
        str(UPLOAD_DIR / analysis.old_image_path),
        str(UPLOAD_DIR / analysis.new_image_path),
    )
    analysis.regions.clear()
    for r in results:
        analysis.regions.append(ChangeRegion(
            object_type=r["object_type"],
            cause_label=r["cause_label"],
            confidence=r["confidence"],
            geometry=json.dumps(r["geometry"]),
        ))
    analysis.status = "completed"
    db.commit()
    return {"id": analysis.id, "status": analysis.status,
            "regions_found": len(results)}


@router.get("/{analysis_id}")
def get_analysis(analysis_id: int, db: Session = Depends(get_db)):
    analysis = db.get(Analysis, analysis_id)
    if not analysis:
        raise HTTPException(404, "Analysis not found")
    return {
        "id": analysis.id,
        "status": analysis.status,
        "created_at": analysis.created_at,
        "regions": [
            {
                "id": r.id,
                "object_type": r.object_type,
                "cause_label": r.cause_label,
                "confidence": r.confidence,
                "geometry": json.loads(r.geometry) if r.geometry else None,
            }
            for r in analysis.regions
        ],
    }