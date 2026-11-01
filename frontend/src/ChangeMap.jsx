import { MapContainer, TileLayer, GeoJSON } from "react-leaflet";
import "leaflet/dist/leaflet.css";

export default function ChangeMap({ regions }) {
  const features = regions
    .filter((r) => r.geometry)
    .map((r) => ({
      type: "Feature",
      geometry: r.geometry,
      properties: {
        object_type: r.object_type,
        cause_label: r.cause_label,
        confidence: r.confidence,
      },
    }));

  if (features.length === 0) return <p>No geographic data to display.</p>;

  const collection = { type: "FeatureCollection", features };

  // centre the map on the first coordinate of the first polygon
  const [lon, lat] = features[0].geometry.coordinates[0][0];

  function onEach(feature, layer) {
    const p = feature.properties;
    layer.bindPopup(
      `<b>${p.object_type}</b><br/>Cause: ${p.cause_label}<br/>Confidence: ${Math.round(
        p.confidence * 100
      )}%`
    );
  }

  return (
    <div style={{ height: 400, marginTop: 16 }}>
      <MapContainer
        center={[lat, lon]}
        zoom={16}
        style={{ height: "100%", width: "100%" }}
      >
        <TileLayer
          attribution="&copy; OpenStreetMap contributors"
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <GeoJSON data={collection} onEachFeature={onEach} />
      </MapContainer>
    </div>
  );
}