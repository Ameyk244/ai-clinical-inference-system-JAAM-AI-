import requests
from PIL import Image
import io
from unittest.mock import patch

URL = "http://127.0.0.1:8080/api/predict"

@patch("app.model_service.predict_ensemble")
def test_valid_image_ci(mock_predict):
    mock_predict.return_value = ("TestLabel", 95.0, None)

    img = Image.new("RGB", (224, 224), color="white")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    response = requests.post(
        URL,
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )

    assert response.status_code == 200

    data = response.json()
    assert data["label"] == "TestLabel"
    assert data["confidence"] == 95.0


def test_no_file_ci():
    response = requests.post(URL)
    assert response.status_code == 400