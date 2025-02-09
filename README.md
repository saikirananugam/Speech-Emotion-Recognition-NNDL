# Speech Emotion Recognition using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-brightgreen)](https://www.tensorflow.org/)
[![Librosa](https://img.shields.io/badge/Librosa-Audio%20Processing-orange)](https://librosa.org/)


## 📌 Overview

Speech Emotion Recognition (SER) is a deep learning-based system that detects human emotions from speech data. It utilizes **Multi-Layer Perceptron (MLP)**, **Convolutional Neural Networks (CNNs)**, and **Long Short-Term Memory (LSTM)** networks for emotion classification.

The model extracts **MFCC, Chroma, and Mel spectrogram** features from audio samples and classifies them into **calm, happy, fearful, and disgust** emotions using the **RAVDESS dataset**.

## Features

✅ Extracts **MFCC, Chroma, and Mel** features  
✅ Multi-class **emotion classification** (Calm, Happy, Fearful, Disgust)  
✅ Supports **MLP, CNN, LSTM** models  
✅ Data **preprocessing and augmentation**  
✅ **Confusion Matrix & Accuracy** visualization  
✅ **Interactive UI for recording & real-time detection**  

---

## 📖 Table of Contents

- [Setup Guide](#setup-guide)
- [Installation & Dependencies](#installation--dependencies)
- [Running the Application](#running-the-application)
- [Usage](#usage)
- [Results](#results)
- [Tools & Technologies](#tools--technologies)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🛠️ Setup Guide

### Installation & Dependencies

#### 1️⃣ Clone the Repository
```bash
git clone https://github.com/saikirananugam/Speech-Emotion-Recognition-NNDL.git
cd Speech-Emotion-Recognition-NNDL
```

2️⃣ Install Required Packages
Ensure you have Python 3.8+, then run:

```bash
pip install -r requirements.txt
```

```bash
pip install librosa soundfile numpy scikit-learn tensorflow keras pandas matplotlib seaborn
```

3️⃣ Download the Dataset
The RAVDESS dataset can be downloaded from Kaggle.
Place the dataset inside the datasets/ directory.

Running the Application
1️⃣ Train the Model

```bash

python train.py
```

2️⃣ Test the Model

```bash

python test.py
```

3️⃣ Predict Emotion from an Audio File
```bash
python predict.py --file path/to/audio.wav
```
Example:

```bash
python predict.py --file datasets/Actor_04/03-01-01-01-01-01-04.wav
```
Usage
Extracts MFCC, Chroma, and Mel features from speech.
Trains MLP, CNN, and LSTM models for classification.
Predicts emotions from real-time or stored audio.
Provides an interactive UI for live emotion detection.

📊 Results
Model Performance
Algorithm	Accuracy (%)
MLP Classifier	90%
CNN	92%
LSTM	96%
Confusion Matrix
The confusion matrix visualizes the model’s predictions vs actual labels.

Training Loss & Accuracy Graphs
Graphs showcasing model convergence, validation accuracy, and loss.

🛠️ Tools & Technologies
Development Tools and Technologies Table
<table> <thead> <tr> <th>Development Area</th> <th>Tools/Technologies</th> <th>Description</th> </tr> </thead> <tbody> <tr> <td rowspan="2"><strong>Frontend Development</strong></td> <td><img src="https://img.shields.io/badge/HTML/CSS-blue.svg" alt="HTML/CSS"></td> <td>For structuring and styling web pages.</td> </tr> <tr> <td><img src="https://img.shields.io/badge/Jinja2-yellow.svg" alt="Jinja2"></td> <td>A templating engine for Python.</td> </tr> <tr> <td rowspan="3"><strong>Backend Development</strong></td> <td><img src="https://img.shields.io/badge/Flask-black.svg" alt="Flask"></td> <td>A lightweight WSGI framework for serving web applications.</td> </tr> <tr> <td><img src="https://img.shields.io/badge/Python-blue.svg" alt="Python"></td> <td>The core programming language used for backend development.</td> </tr> <tr> <td><img src="https://img.shields.io/badge/Pandas-purple.svg" alt="Pandas"></td> <td>Essential for data manipulation and analysis.</td> </tr> <tr> <td rowspan="2"><strong>Machine Learning</strong></td> <td><img src="https://img.shields.io/badge/Scikit--Image-orange.svg" alt="Scikit-Image"></td> <td>Used for predictive modeling.</td> </tr> <tr> <td><img src="https://img.shields.io/badge/NumPy-darkblue.svg" alt="NumPy"></td> <td>Supports high-level mathematical functions.</td> </tr> </tbody> </table>


Acknowledgments
Special Thanks to:

SR Engineering College & SR University for research support
Contributors: Saikiran Anugam
The creators of the RAVDESS Dataset
Librosa & TensorFlow Developers for their amazing tools


