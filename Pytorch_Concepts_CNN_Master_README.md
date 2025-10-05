# 🧠 PyTorch Concepts, CNN, and Hands-on Projects

*A complete practical repository to learn PyTorch fundamentals, convolutional neural networks (CNNs), custom dataset loading, and real-world deep learning applications.*

---

## 🌍 Overview

This repository is a **structured learning resource** designed to help you build strong foundations in **PyTorch**, **computer vision**, and **deep learning**. It combines theoretical notebooks, real-world projects, and research references.

The collection spans from **tensor manipulation basics** to **custom dataset creation**, **CNN model training**, and specialized projects such as **Alzheimer’s Disease Prediction** and **Facial Expression Recognition**.

---

## 🗂️ Repository Structure

```
Pytorch concept, CNN and hands-on/
│
├── Alzheimer's Disease Prediction/
│   ├── Dataset/
│   ├── Images/
│   ├── Models/
│   ├── Web-App/
│   ├── Requirements.txt
│
├── CNN-Training-Using-a-Custom-Dataset/
│   ├── 016-CNN-Training-Using-a-Custom-Dataset.ipynb
│   ├── 016-cnn_model.pth
│   ├── KNCVU-Computer_Vision-Pytorch-Dataset_resource.txt
│
├── Defining-custom-Image-Dataset-loader-and-usage/
│   ├── Custom Dataset definition notebooks & PDFs
│
├── Facial_Expression_Recognition_with_PyTorch/
│   ├── Emotion detection model notebooks
│
├── Core PyTorch Tutorials/
│   ├── 002-Introduction-to-tensors.ipynb
│   ├── 003-Indexing-Tensors.ipynb
│   ├── 004-Using-Random-Numbers-to-create-noise-image.ipynb
│   ├── 005-Tensors-of-Zero_s-and-One_s.ipynb
│   ├── 006-Tensor DataTypes.ipynb
│   ├── 007-Tensor_Manipulation.ipynb
│   ├── 010-Stack-Operation.pdf
│   ├── 011-Understanding-Pytorch-neural-network-components.ipynb
│   ├── 012-Create-Linear-Regression-model-with-Pytorch-components.ipynb
│   ├── 013-Multi-Class-classification-with-pytorch-using-custom-neural-networks.ipynb
│   ├── 014-Understanding-components-of-custom-data-loader-in-pytorch.ipynb
│   ├── 015-Defining-custom-Image-Dataset-loader-and-usage.pdf
│   ├── 016-CNN-Training-Using-a-Custom-Dataset.pdf
│
└── README.md (this master file)
```

---

## 🔍 Learning Modules

### 1️⃣ PyTorch Fundamentals
Core building blocks for working with tensors and building neural networks.

**Topics Covered**
- Tensor creation, indexing, slicing, and reshaping  
- Mathematical operations on tensors  
- Broadcasting and gradients  
- Linear regression from scratch using `torch.nn`  
- Building custom training loops  

**Key Notebooks:**  
`002-Introduction-to-tensors.ipynb`, `006-Tensor DataTypes.ipynb`, `007-Tensor_Manipulation.ipynb`

---

### 2️⃣ Neural Network Components
Detailed exploration of layers, activations, optimizers, and loss functions in PyTorch.

**Concepts**
- Using `torch.nn.Module` and forward propagation  
- Implementing backpropagation manually  
- Activation functions (ReLU, Sigmoid, Tanh)  
- Loss functions and optimizers (SGD, Adam)  
- Custom network components  

**Notebook:** `011-Understanding-Pytorch-neural-network-components.ipynb`

---

### 3️⃣ Building Custom Datasets and Dataloaders
Shows how to create and load custom image datasets efficiently for training deep models.

**Concepts**
- Using `torch.utils.data.Dataset` and `DataLoader`  
- Custom transformations with `torchvision.transforms`  
- Batch loading and memory-efficient pipelines  

**Notebook:** `014-Understanding-components-of-custom-data-loader-in-pytorch.ipynb`

---

### 4️⃣ CNN Training with Custom Datasets
Step-by-step CNN training pipeline using your own dataset.

**Highlights**
- Creating a dataset folder structure  
- Implementing CNN architecture using `torch.nn.Conv2d`, `MaxPool2d`, `Linear` layers  
- Forward & backward pass with training/validation  
- Saving and loading trained models (`.pth` format)  
- Evaluation metrics: accuracy, loss curves  

**Notebook:** `016-CNN-Training-Using-a-Custom-Dataset.ipynb`

---

### 5️⃣ Facial Expression Recognition with PyTorch
A full CNN model trained on facial emotion data to classify expressions such as happy, sad, angry, etc.

**Features**
- Preprocessing with OpenCV  
- CNN feature extraction + dense classification layers  
- Data augmentation (rotation, flip, normalization)  
- Evaluation via confusion matrix and F1 score  

---

### 6️⃣ Alzheimer’s Disease Prediction
End-to-end classification of MRI brain scans to detect Alzheimer’s disease stages.

**Pipeline**
- Dataset preparation and preprocessing  
- CNN model training for image classification  
- Web application integration (Flask) for live predictions  

**Directories**
- `Dataset/` → Image data (MRI scans)  
- `Models/` → Trained weights (.pth)  
- `Web-App/` → Deployment interface  

---

## 🧩 Example Workflow

### Train a Custom CNN
```bash
cd "CNN-Training-Using-a-Custom-Dataset"
jupyter notebook "016-CNN-Training-Using-a-Custom-Dataset.ipynb"
```

### Launch Alzheimer’s Disease Predictor
```bash
cd "Alzheimer's Disease Prediction/Web-App"
pip install -r Requirements.txt
python app.py
```

---

## ⚙️ Requirements

Install dependencies with:
```bash
pip install torch torchvision torchaudio matplotlib numpy pandas scikit-learn opencv-python tqdm
```

---

## 🎓 Learning Objectives

By working through this repository, you will:
- Understand **PyTorch tensor fundamentals**  
- Build and train **neural networks and CNNs** from scratch  
- Create **custom datasets and dataloaders**  
- Apply CNNs to **real-world applications** (Alzheimer’s & facial emotion recognition)  
- Gain end-to-end experience from **data preprocessing to model deployment**  

---

## 🧠 Acknowledgments
Special Thanks to **[Krish Naik](https://github.com/krishnaik06)** for the educational guidance and inspiration for multiple PyTorch modules and implementations.

---

## 🧑‍💻 Author  
**Pankaj Somkuwar**  
🔗 [GitHub](https://github.com/Pankaj-Leo) | [LinkedIn](https://linkedin.com/in/pankajsomkuwar)

---

## 🏁 License  
Released under the **MIT License** — open for learning, research, and academic use.

---


