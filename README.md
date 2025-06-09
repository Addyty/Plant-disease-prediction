<<<<<<< HEAD

# 🌿 Plant Disease Prediction using ResNet-50 (PyTorch)

This project provides a deep learning solution to detect and classify plant diseases from leaf images using the **ResNet-50** architecture. Built with PyTorch, it covers the complete workflow from training to deployment.

---

## 🚀 Features

- ✅ Preprocessing & data augmentation with **Torchvision Transforms**
- 🧠 Training with **ResNet-50** and **transfer learning**
- 📉 Evaluation with **confusion matrix** and **classification report**
- 🖼️ Inference pipeline for new/unseen images
- 🌐 Flask API (`app.py`) for real-time deployment

---

## 🧾 Project Structure

```
Plant-disease-prediction-main/
├── train_model.py           # Script to train and save the ResNet-50 model
├── app.py                   # Flask app for deployment
├── class_names.txt          # List of class names
├── train/                   # Training images (in subfolders per class)
├── test/                    # Test images (in subfolders per class)
├── README.md                # Project documentation
```

---

## 🔧 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/plant-disease-prediction.git
cd plant-disease-prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up your dataset**
Place your dataset in the following structure:
```
.
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
├── test/
    ├── class1/
    ├── class2/
    └── ...
=======
# Plant Disease Classifier 🌱

A deep learning web application for automatic plant disease detection from leaf images. Upload or drag-and-drop a plant leaf image and instantly get a prediction of the disease class using a trained ResNet50 model.

---

## 🚀 Project Overview

This project helps farmers, researchers, and agriculturalists identify plant diseases from images of leaves. It uses a deep learning model (ResNet50) trained on the PlantDoc dataset and provides a user-friendly web interface for predictions.

---

## ✨ Features
- **Image Upload & Drag-and-Drop:** Upload or drag a leaf image to get instant predictions.
- **Modern Web UI:** Clean, responsive, and professional frontend.
- **FastAPI Backend:** Efficient Python backend serving the trained PyTorch model.
- **Custom Training:** Easily retrain the model on your own data.
- **Multi-class Support:** Handles many plant species and disease types.

---

## 🖼️ How It Works
1. **Train the Model:** Use your dataset to train a ResNet50 model. The best model and class names are saved.
2. **Start the Backend:** FastAPI serves a `/predict/` endpoint for image classification.
3. **Use the Web App:** Open the frontend, upload an image, and see the prediction.

---

## 🛠️ Setup Instructions


```

### 1. Prepare the Dataset
- Organize your dataset as:
  ```
  train/
    Class1/
      img1.jpg
      ...
    Class2/
      ...
  test/
    Class1/
      ...
    Class2/
      ...
  ```
- (Or use the [PlantDoc Dataset](https://github.com/pratikkayal/PlantDoc-Dataset.git))

### 2. Train the Model
```bash
python train_model.py train
```
- This will save `best_plant_disease_model.pth` and `class_names.txt`.

### 3. Start the Backend
```bash
uvicorn app:app --reload
```

### 4. Launch the Frontend
- Open `index.html` in your browser (double-click or right-click > Open with browser).

---

## 🧑‍💻 Usage

### **Train the Model**
```
python train_model.py train
```

### **Predict from Command Line**
```
python train_model.py predict path/to/image.jpg
```

### **Web App**
- Open `index.html` in your browser.
- Drag & drop or select an image.
- See the prediction instantly!

---

## 🖥️ Technologies Used
- **PyTorch** (Deep Learning)
- **FastAPI** (Backend API)
- **HTML/CSS/JS** (Frontend)

---

## 📁 Folder Structure
```
├── app.py                # FastAPI backend
├── index.html            # Frontend web app
├── train_model.py        # Model training & CLI prediction
├── best_plant_disease_model.pth  # Saved model weights (after training)
├── class_names.txt       # Saved class names (after training)
├── train/                # Training images (one folder per class)
├── test/                 # Test images (one folder per class)
>>>>>>> 6f1eda5 (Initial commit)
```

---

<<<<<<< HEAD
## 🧠 Model Training

Run the training script:
```bash
python train_model.py
```

The best model will be saved as `resnet50_best.pth`.

---

## 🔍 Inference Example

```python
from PIL import Image
image = Image.open("test/Apple___Apple_scab/sample.jpg")
predict_image(image, model, data_transforms['test'], class_names)
```

---

## 📈 Evaluation

- Accuracy
- Confusion matrix
- Precision, recall, F1-score per class

---

## 🌐 Deployment (Optional)

Use `Flask` to deploy the model as an API:

```bash
python app.py
```

Send POST requests with leaf images and get back predicted class labels.

---

## 📚 Libraries Used

- Python 3.8+
- PyTorch
- Torchvision
- Scikit-learn
- Flask
- Matplotlib
- Seaborn

---

## 📌 License

This project is licensed under the MIT License. Feel free to use and adapt it for academic or research purposes.

---

## 🤝 Contribution

Pull requests and suggestions are welcome!

---

## ✍️ Author

**Aaditya Tyagi**  
MSc Data Analytics – Berlin School of Business and Innovation
=======

**Made with ❤️ for the agricultural community.**
>>>>>>> 6f1eda5 (Initial commit)
