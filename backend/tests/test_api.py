def upload(client, png_bytes):
    return client.post("/analyses", files={
        "old_image": ("a.png", png_bytes, "image/png"),
        "new_image": ("b.png", png_bytes, "image/png"),
    })


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_upload_ok(client, png_bytes):
    r = upload(client, png_bytes)
    assert r.status_code == 201
    assert r.json()["status"] == "pending"


def test_upload_rejects_wrong_extension(client, png_bytes):
    r = client.post("/analyses", files={
        "old_image": ("notes.txt", b"hello", "text/plain"),
        "new_image": ("b.png", png_bytes, "image/png"),
    })
    assert r.status_code == 400


def test_upload_rejects_fake_image(client, png_bytes):
    r = client.post("/analyses", files={
        "old_image": ("fake.png", b"not an image", "image/png"),
        "new_image": ("b.png", png_bytes, "image/png"),
    })
    assert r.status_code == 400


def test_get_missing_returns_404(client):
    assert client.get("/analyses/9999").status_code == 404


def test_run_and_get(client, png_bytes):
    aid = upload(client, png_bytes).json()["id"]
    run = client.post(f"/analyses/{aid}/run")
    assert run.status_code == 200
    assert run.json()["status"] == "completed"
    data = client.get(f"/analyses/{aid}").json()
    assert data["status"] == "completed"
    assert len(data["regions"]) >= 1
    assert data["regions"][0]["cause_label"] in ("human", "natural", "unknown")


def test_list_and_stats(client, png_bytes):
    aid = upload(client, png_bytes).json()["id"]
    client.post(f"/analyses/{aid}/run")
    assert len(client.get("/analyses").json()) == 1
    assert client.get("/analyses/stats/summary").json()["total_analyses"] == 1


def test_report_is_pdf(client, png_bytes):
    aid = upload(client, png_bytes).json()["id"]
    client.post(f"/analyses/{aid}/run")
    r = client.get(f"/analyses/{aid}/report")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content.startswith(b"%PDF")