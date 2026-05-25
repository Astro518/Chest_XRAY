# 🫁 Chest X-Ray Pneumonia Detection using ResNet50

A deep learning model that automatically detects pneumonia from chest X-ray images using transfer learning with ResNet50. This project achieves high classification accuracy through systematic hyperparameter tuning, class imbalance handling, and comprehensive data augmentation.

## 📊 Dataset
- **Source**: Chest X-Ray Images (Pneumonia) from Kaggle
- **Classes**: Normal (healthy) vs. Pneumonia (infected)
- **Split**: Train / Validation / Test directories

## 🧠 Model Architecture
- **Backbone**: ResNet50 pre-trained on ImageNet
- **Custom Head**: Modified fully connected layer for binary classification
- **Loss Function**: Cross-entropy with class weights to handle imbalance
- **Optimization**: Adam optimizer with StepLR scheduler

## 🔧 Key Features

### Data Preprocessing
- Resizing to 224×224 pixels (ResNet50 input size)
- Normalization using ImageNet mean & std
- Data augmentation: Random horizontal flip, rotation (±15°), color jitter

### Hyperparameter Tuning
Grid search over:
- **Batch sizes**: 16, 32
- **Learning rates**: 1e-4, 1e-3  
- **Weight decays**: 0, 1e-4

### Performance Metrics
- Accuracy score
- Classification report (precision, recall, F1-score)
- Confusion matrix

## 🚀 Speed Optimizations
- `cudnn.benchmark = True` for CUDA optimization
- GPU acceleration support
- Multi-worker data loading

## 🛠️ Technologies Used
- **Framework**: PyTorch & TorchVision
- **Model**: ResNet50 (transfer learning)
- **Metrics**: scikit-learn
- **Hardware**: CUDA-enabled GPU recommended

## 📈 Sample Output
