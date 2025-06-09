
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
```

---

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
