import { useState } from "react";

export default function CompareSlider({ before, after }) {
  const [pos, setPos] = useState(50);

  return (
    <div className="compare">
      <img src={after} alt="After" className="compare-img" />
      <div
        className="compare-before"
        style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}
      >
        <img src={before} alt="Before" className="compare-img" />
      </div>
      <div className="compare-line" style={{ left: `${pos}%` }} />
      <span className="tag tag-l">BEFORE</span>
      <span className="tag tag-r">AFTER</span>
      <input
        type="range"
        min="0"
        max="100"
        value={pos}
        onChange={(e) => setPos(Number(e.target.value))}
        className="compare-range"
        aria-label="Drag to compare before and after"
      />
    </div>
  );
}