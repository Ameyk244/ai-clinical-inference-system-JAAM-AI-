from tensorflow.keras.models import load_model

# VGG19
model = load_model("vgg19.h5", compile=False)
model.save("vgg19_fixed.keras")

# ResNet
model = load_model("resnet.keras", compile=False)
model.save("resnet_fixed.keras")

# DenseNet
model = load_model("densenet.keras", compile=False)
model.save("densenet_fixed.keras")

print("✅ All models fixed and saved")