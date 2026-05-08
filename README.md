
# JAAM AI: Pioneering Early Alzheimer's Detection


# Overview

JAAMNET is an ensemble deep learning framework developed for the automated detection and staging of Alzheimer’s disease using MRI brain scans. Alzheimer’s disease is a progressive neurodegenerative condition characterized by memory loss, cognitive decline, and structural brain atrophy. Early and accurate identification of disease severity can significantly enhance treatment planning and patient management.

To ensure robustness and generalization, JAAMNET was trained on two large-scale MRI datasets:

## 40K Augmented Alzheimer MRI Dataset V2 (Axial MRI slices)

## 80K OASIS Dataset

Together, these datasets encompass four diagnostic classes — Non-Demented, Very Mild Demented, Mild Demented, and Moderate Demented — allowing the framework to classify varying stages of cognitive decline with high precision.

JAAMNET utilizes an ensemble learning architecture that combines multiple deep learning backbones — DenseNet201, ResNet50V2, VGG19, and a custom CNN — through a voting-based system. This ensemble strategy integrates the strengths of each model, improving classification accuracy, reducing overfitting, and enhancing generalization across diverse MRI scans.

Among the individual models, DenseNet201 achieved 99.6% val accuracy, followed by ResNet50V2 (99.3%) and VGG19 (98.4%). The voting ensemble further stabilized predictions, delivering consistent performance across datasets and Alzheimer’s severity classes.

In addition to performance optimization, JAAMNET emphasizes interpretability through Grad-CAM (Gradient-weighted Class Activation Mapping), which highlights critical brain regions influencing model predictions. This interpretability ensures clinical transparency and helps validate that the model focuses on neurologically relevant areas.

## Demo

link to demo

https://drive.google.com/file/d/1BVF6eiLXBY45HCtoLaWU1xuixukiGfAm/view?usp=sharing



# Methodology

### Data Preprocessing

MRI brain images are normalized, resized, and augmented for training stability.

### Model Training

Transfer learning applied on pre-trained ImageNet weights.

Fine-tuning strategies used for DenseNet201, ResNet50V2, and VGG19.

Adam optimizer with learning rate scheduling and regularization.

### Evaluation Metrics

Accuracy, precision, recall, and F1-score evaluated on validation data.

Grad-CAM visualizations generated for interpretability assessment.

### Deployment

Flask-based interface for MRI upload and automated model inference.
## Screenshots

![App Screenshot](https://dl.dropboxusercontent.com/scl/fi/tu1c3yro8bbt2gnvrra3l/Screenshot-2025-11-04-115438.png?rlkey=md4g6c4kikuvit3fb36w7iglp)

![App Screenshot](https://dl.dropboxusercontent.com/scl/fi/4m4hsxr7cex5h8atptjsu/Screenshot-2025-11-04-104521.png?rlkey=1atfibu4ilmrwp1h4h3lh6wzs)

![App Screenshot](https://dl.dropboxusercontent.com/scl/fi/6co4sdn2orwm93kbf8sdz/Screenshot-2025-11-04-104614.png?rlkey=0dhbwkvv5xp81qonku9iyu7zs)






## Tech Stack

**Programming Language:** Python

**Deep Learning Frameworks:** TensorFlow, Keras

**Supporting Libraries:** NumPy, Pandas, Matplotlib, OpenCV, Scikit-learn, Seaborn

**Model Architectures:** DenseNet201, ResNet50V2, VGG19, Custom CNN

**Training & Development Environment:** Kaggle (GPU P100)

**Optimization Techniques:** Transfer Learning, Fine-tuning, Learning Rate Scheduling, Early Stopping

**Visualization & Explainability:** Grad-CAM, Heatmap Visualization
Web Deployment: Flask Framework

**Version Control & Tools:** Git, Docker


