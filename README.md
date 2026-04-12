# Deep_Learning_Brain_Tumor_Classification


## Project Information:
**Project Title**: Comparison of Pretrained CNN Models and Hidden Layer Architectures for Brain Tumor Classification Using MRI Images

**Model**: Pretrained CNN models (MobileNetV2, ResNet50, EfficientNetB0) and hidden layers architectures (triangular, recgtangular, recto-triangular)


**Project Description**:
This project aims to classify brain tumor MRI images into four categories: glioma, meningioma, pituitary, and no tumor using a deep learning approach. A total of 9 combinations of pretrained CNN models and hidden layer architecture were tested out to determine which is the best combination for the task.
The dataset consists of labeled MRI images, which were preprocessed through resizing, normalization, and augmentation techniques. The model was trained and evaluated using multiple performance metrics including accuracy, precision, recall, and F1-score.
Experimental results show that the proposed model achieves strong classification performance, with the best configuration reaching approximately **82.49% accuracy**. Visualization techniques such as confusion matrix and training curves were used to analyze model behavior.


**Objective**:
- Develop a CNN model for brain tumor classification
- Compare multiple combination of pretrained models and architecture variations
- Evaluate model performance using standard metrics
- Analyze strengths and weaknesses of the model


**Dataset**:
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
Type: Images
Classes:
- No tumor
- Glioma
- Meningioma
- Pituitary

Dataset not included due to its size


**Dependencies**:
All the depencies needed are documented in requirements.txt


**Inference** (Using trained model):

#Load model
from tensorflow.keras.models import load_model 
model = load_model("best_model.h5")

#Prepare image
import numpy as np 
from tensorflow.keras.preprocessing import image 
img = image.load_img("test.jpg", target_size=(224, 224)) 
img_array = image.img_to_array(img) / 255.0 
img_array = np.expand_dims(img_array, axis=0)

#Predict
prediction = model.predict(img_array) 
class_names = ["glioma", "meningioma", "notumor", "pituitary"] 
predicted_class = class_names[np.argmax(prediction)] 
confidence = np.max(prediction) 
print(predicted_class, confidence)


**Model Comparison**:

<img width="1200" height="600" alt="model_comparison" src="https://github.com/user-attachments/assets/9b0d5cba-bfbe-41c8-a489-3b50ca95193a" />

Insight:
MobileNetV2 + triangular architecture achieve the best results and outperforms other model combinations across all metrics

**Training History**:

<img width="1200" height="500" alt="training_history" src="https://github.com/user-attachments/assets/681c5459-abb4-4766-b9d9-a7f7ccb018be" />

Insight:
Training and validation accuracy increase steadily but there is slight overfitting in later epochs

**Results**:
Classification report
| Class        | Precision | Recall | F1-Score | Support |
|-------------|----------|--------|----------|---------|
| Glioma      | 0.93     | 0.63   | 0.75     | 400     |
| Meningioma  | 0.74     | 0.69   | 0.72     | 400     |
| No Tumor    | 0.93     | 0.98   | 0.95     | 400     |
| Pituitary   | 0.75     | 0.99   | 0.85     | 400     |
| **Accuracy** |          |        | **0.82** | 1600    |
| Macro Avg | 0.84   | 0.82   | 0.82     | 1600    |
| Weighted Avg | 0.84 | 0.82 | 0.82     | 1600    |


**Confusion Matrix**:

<img width="800" height="600" alt="confusion_matrix" src="https://github.com/user-attachments/assets/f69e2975-1a5f-43a1-a771-b207704a0773" />

Insight:
Strong performance in "pituitary" and "notumor" classes, but missclassifications occur between "glioma" and "meningioma" classes


**Model Weights**:
outputs/models/best_model.h5

Author: Callista Serena Ekaputri
