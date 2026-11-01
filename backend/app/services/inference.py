def run_inference(old_path: str, new_path: str) -> list[dict]:
    """STUB: dummy output. ML teammate replaces this body with the real model.

    Contract: return a list of dicts with keys
      object_type: "building" | "road" | "vehicle"
      cause_label: "human" | "natural" | "unknown"
      confidence:  float 0..1
      geometry:    GeoJSON geometry dict (lon, lat)
    """
    return [
        {
            "object_type": "building",
            "cause_label": "unknown",
            "confidence": 0.5,
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [75.860, 22.720], [75.861, 22.720],
                    [75.861, 22.721], [75.860, 22.721],
                    [75.860, 22.720],
                ]],
            },
        }
    ]