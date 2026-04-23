import sys
import os

# ✅ Fix Python path so 'app' module is found
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import json
from app.model_service import predict_ensemble

# ✅ Path to your real sample image
IMAGE_PATH = os.path.join(BASE_DIR, "sample", "test.jpg")


def main():
    try:
        if not os.path.exists(IMAGE_PATH):
            raise FileNotFoundError(f"{IMAGE_PATH} not found")

        print(f"📸 Running inference on: {IMAGE_PATH}")

        label, conf, gradcam = predict_ensemble(IMAGE_PATH)

        result = {
            "label": label,
            "confidence": conf,
            "gradcam": gradcam
        }

        # ✅ Create output folder
        output_dir = os.path.join(BASE_DIR, "ci_results")
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, "result.json")

        with open(output_path, "w") as f:
            json.dump(result, f, indent=4)

        print("✅ Inference complete")
        print(f"📄 Result saved at: {output_path}")

    except Exception as e:
        print("🔥 ERROR during inference:", str(e))
        raise


if __name__ == "__main__":
    main()