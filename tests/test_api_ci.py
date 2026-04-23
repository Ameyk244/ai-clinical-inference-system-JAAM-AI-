import io
from PIL import Image
from unittest.mock import patch
import main

def test_valid_image_ci():
    client = main.app.test_client()

    img = Image.new("RGB", (224, 224), color="white")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    with patch("main.predict_ensemble") as mock_predict:
        mock_predict.return_value = ("TestLabel", 95.0, None)

        response = client.post(
            "/api/predict",
            data={"file": (img_bytes, "test.jpg")},
            content_type="multipart/form-data"
        )

    assert response.status_code == 200
    data = response.get_json()

    assert data["label"] == "TestLabel"
    assert data["confidence"] == 95.0


def test_no_file_ci():
    client = main.app.test_client()

    response = client.post("/api/predict")
    assert response.status_code == 400