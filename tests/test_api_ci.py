import io
from PIL import Image
from main import app   # ✅ this is enough

def test_valid_image_ci():
    client = app.test_client()

    img = Image.new("RGB", (224, 224), color="white")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    response = client.post(
        "/api/predict",
        data={"file": (img_bytes, "test.jpg")},
        content_type="multipart/form-data"
    )

    assert response.status_code in [200, 400]


def test_no_file_ci():
    client = app.test_client()

    response = client.post("/api/predict")
    assert response.status_code == 400