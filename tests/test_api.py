import requests

URL = "http://127.0.0.1:8080/api/predict"

def test_valid_image():
    with open("uploads/27.jpg", "rb") as f:
        response = requests.post(URL, files={"file": f})
    
    assert response.status_code == 200
    data = response.json()
    
    assert "label" in data
    assert "confidence" in data

def test_no_file():
    response = requests.post(URL)
    assert response.status_code == 400