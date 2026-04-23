import os
from main import app

# 🔥 lazy loading patch (only for Render)
from app import model_service

_original_predict = model_service.predict_ensemble
_models_loaded = False

def lazy_predict(img_path):
    global _models_loaded

    if not _models_loaded:
        print("🚀 Loading models (first request)...")
        _models_loaded = True

    return _original_predict(img_path)

# override only in this runtime
model_service.predict_ensemble = lazy_predict


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Running Render app on port {port}")
    app.run(host="0.0.0.0", port=port)