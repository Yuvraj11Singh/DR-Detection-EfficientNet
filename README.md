# 🩺 Diabetic Retinopathy Detection using EfficientNet

An AI-based medical image classification project that detects **Diabetic Retinopathy (DR)** from retinal fundus images using **EfficientNetB0** and TensorFlow.

This project demonstrates how deep learning can assist in early screening of diabetic retinopathy and support scalable healthcare diagnostics.

---

## 📌 Overview
Diabetic Retinopathy is a leading cause of blindness worldwide. Early detection through retinal screening can prevent severe vision loss. This project builds a deep learning model that automatically classifies retinal fundus images into DR and No-DR categories.

---

## 🗂 Dataset 

**Dataset Used:**  
APTOS 2019 Blindness Detection  
Source: Kaggle  
https://www.kaggle.com/competitions/aptos2019-blindness-detection

The APTOS dataset contains high-resolution retinal fundus images labeled based on the severity of diabetic retinopathy.

For this project, the original multi-class labels were simplified into a **binary classification problem**:

- Class 0 → No Diabetic Retinopathy  
- Class 1 → Diabetic Retinopathy present  

The dataset was preprocessed and split into training and validation sets using TensorFlow’s data pipeline.

### Data Processing
- Images resized to 224×224  
- Batch size: 64  
- Train-validation split: 80/20  
- Loaded using `image_dataset_from_directory()`  

> Note: The dataset is not uploaded to this repository due to size constraints. Please download it from Kaggle and organize it into class-wise folders before running the notebook.

---

## 🧠 Model Architecture and Method Implemented 

- Base model: EfficientNetB0 (pretrained on ImageNet)
- Global Average Pooling layer
- Dropout (0.4)
- Dense sigmoid output layer

### Training Strategy
1. Freeze base model and train classifier head  
2. Unfreeze top layers for fine-tuning  
3. Train with lower learning rate  

Optimizer: Adam  
Loss Function: Binary Crossentropy  

---

## 📊 Results

### Classification Report
```
precision    recall  f1-score   support

0       0.95     0.94     0.94      385
1       0.94     0.95     0.95      414

accuracy                        0.94      799
macro avg   0.94     0.94     0.94      799
weighted avg 0.94    0.94     0.94      799
```

### Final Metrics
- **Accuracy:** 94%  
- **AUC Score:** 0.985  
- **F1 Score:** 0.94  

---

## 🛠 Tech Stack
- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- Scikit-learn  
- Google Colab  

---

## 🚀 How to Run ( Steps to execute )

```
git clone https://github.com/YOUR_USERNAME/dr-detection-efficientnet.git
cd dr-detection-efficientnet
pip install tensorflow numpy matplotlib scikit-learn
```

Download dataset from Kaggle and organize into class folders, then run:
```
dr_detection.ipynb
```

---

## 🔬 Future Improvements
- Multi-class DR severity detection  
- Larger dataset training  
- Model deployment  
- Web app integration  

---

## 👨‍💻 Author
**Yuvraj Singh**  

---

## 📜 License
For academic and research use only.
