from datetime import datetime, timezone
from sqlalchemy import String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .database import Base

def now():
    return datetime.now(timezone.utc)

class Analysis(Base):
    __tablename__ = "analyses"
    id: Mapped[int] = mapped_column(primary_key=True)
    old_image_path: Mapped[str] = mapped_column(String(255))
    new_image_path: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(20), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    regions: Mapped[list["ChangeRegion"]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )

class ChangeRegion(Base):
    __tablename__ = "change_regions"
    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_id: Mapped[int] = mapped_column(ForeignKey("analyses.id"), index=True)
    object_type: Mapped[str] = mapped_column(String(20))
    cause_label: Mapped[str] = mapped_column(String(20), default="unknown")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    geometry: Mapped[str | None] = mapped_column(Text)
    analysis: Mapped["Analysis"] = relationship(back_populates="regions")