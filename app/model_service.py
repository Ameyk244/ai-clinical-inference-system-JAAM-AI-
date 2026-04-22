import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from huggingface_hub import hf_hub_download
from flask import Flask, request, render_template_string, send_from_directory, url_for,jsonify
from PIL import Image
from dotenv import load_dotenv
import cv2

# ----------------------------
# Env + Config
# ----------------------------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_REPO = "subx24/ml-alzheimer-models"

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

CLASS_LABELS = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
CONFIDENCE_THRESHOLD = 0.7

# ----------------------------
# Load Models
# ----------------------------
def load_models():
    model_files = {
        "vgg19": "vgg19.h5",
        "resnet": "resnet.keras",
        "densenet": "densenet.keras"
    }
    models = {}
    for name, fname in model_files.items():
        path = hf_hub_download(MODEL_REPO, fname, token=HF_TOKEN)
        models[name] = load_model(path)
        print(f"✅ Loaded {name} from {fname}")
    return models

MODELS = load_models()

# ----------------------------
# Preprocessing
# ----------------------------
def preprocess(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    arr = image.img_to_array(img)/255.0
    arr = np.expand_dims(arr,0)
    return img, arr

# ----------------------------
# Single prediction
# ----------------------------
def predict_single(img_array, model_name):
    model = MODELS[model_name]
    preds = model.predict(img_array)
    idx = int(np.argmax(preds))
    conf = float(np.max(preds))
    label = CLASS_LABELS[idx] if conf >= CONFIDENCE_THRESHOLD else "Unknown / Invalid MRI"
    return label, conf*100, idx

# ----------------------------
# Grad-CAM (for VGG19)
# ----------------------------
def get_grad_cam(model, img_array, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        # Make sure predictions is a tensor
        if isinstance(predictions, list):
            predictions = predictions[0]

        predictions = tf.convert_to_tensor(predictions)

        # pick the correct class
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        # safe indexing: take first batch
        class_channel = predictions[0, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(img_path, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, colormap)
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return superimposed_img

def save_gradcam(img_path, class_index=None):
    _, img_array = preprocess(img_path)
    heatmap = get_grad_cam(MODELS['vgg19'], img_array, last_conv_layer_name="block5_conv4", pred_index=class_index)
    overlay = overlay_heatmap(img_path, heatmap)
    filename = os.path.basename(img_path)
    save_path = os.path.join(RESULTS_FOLDER, f"gradcam_{filename}")
    cv2.imwrite(save_path, overlay)
    return save_path

# ----------------------------
# Ensemble prediction
# ----------------------------
def predict_ensemble(img_path):
    _, arr = preprocess(img_path)
    results = [predict_single(arr, m) for m in MODELS]
    labels, confs, idxs = zip(*results)

    valid = [l for l in labels if l!="Unknown / Invalid MRI"]
    if len(valid)!=len(MODELS) or len(set(valid))!=1:
        return "Unknown / Invalid MRI",0.0,None
    class_index = idxs[0]
    gradcam_path = save_gradcam(img_path, class_index)
    return valid[0], np.mean([c for l,c in zip(labels,confs) if l==valid[0]]), gradcam_path

