import io
import json
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import ALLOWED_EXT, MAX_FILE_MB, UPLOAD_DIR
from ..database import get_db
from ..models import Analysis, ChangeRegion
from ..services.inference import run_inference
from ..services.report import build_report

router = APIRouter(prefix="/analyses", tags=["analyses"])


def check_date(value: str | None, label: str) -> str | None:
    if not value:
        return None
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(400, f"{label} must be in YYYY-MM-DD format")
    return value


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
    location_name: str | None = Form(None),
    old_date: str | None = Form(None),
    new_date: str | None = Form(None),
    db: Session = Depends(get_db),
):
    location_name = (location_name or "").strip() or None
    if location_name and len(location_name) > 120:
        raise HTTPException(400, "Location name is too long (max 120)")
    old_date = check_date(old_date, "Old date")
    new_date = check_date(new_date, "New date")
    if old_date and new_date and old_date > new_date:
        raise HTTPException(400, "Old image date must not be after new image date")

    old_name = await save_image(old_image)
    new_name = await save_image(new_image)
    analysis = Analysis(
        old_image_path=old_name,
        new_image_path=new_name,
        location_name=location_name,
        old_date=old_date,
        new_date=new_date,
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    return {"id": analysis.id, "status": analysis.status}


@router.get("")
def list_analyses(db: Session = Depends(get_db)):
    rows = db.scalars(
        select(Analysis).order_by(Analysis.id.desc()).limit(50)
    ).all()
    return [
        {
            "id": a.id,
            "status": a.status,
            "created_at": a.created_at,
            "location_name": a.location_name,
            "old_date": a.old_date,
            "new_date": a.new_date,
            "regions_count": len(a.regions),
        }
        for a in rows
    ]


@router.get("/stats/summary")
def stats_summary(db: Session = Depends(get_db)):
    total = db.scalar(select(func.count(Analysis.id))) or 0
    regions = db.scalar(select(func.count(ChangeRegion.id))) or 0
    by_type = db.execute(
        select(ChangeRegion.object_type, func.count())
        .group_by(ChangeRegion.object_type)
    ).all()
    by_cause = db.execute(
        select(ChangeRegion.cause_label, func.count())
        .group_by(ChangeRegion.cause_label)
    ).all()
    return {
        "total_analyses": total,
        "total_regions": regions,
        "by_object_type": {k: v for k, v in by_type},
        "by_cause": {k: v for k, v in by_cause},
    }


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


@router.get("/{analysis_id}/report")
def download_report(analysis_id: int, db: Session = Depends(get_db)):
    analysis = db.get(Analysis, analysis_id)
    if not analysis:
        raise HTTPException(404, "Analysis not found")
    pdf = build_report(analysis)
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition":
                 f'attachment; filename="report_{analysis.id}.pdf"'},
    )


@router.get("/{analysis_id}")
def get_analysis(analysis_id: int, db: Session = Depends(get_db)):
    analysis = db.get(Analysis, analysis_id)
    if not analysis:
        raise HTTPException(404, "Analysis not found")
    return {
        "id": analysis.id,
        "status": analysis.status,
        "location_name": analysis.location_name,
        "old_date": analysis.old_date,
        "new_date": analysis.new_date,
        "old_image": f"/files/{analysis.old_image_path}",
        "new_image": f"/files/{analysis.new_image_path}",
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