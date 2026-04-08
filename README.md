# 🎙️ Indian Language Identification from Audio

A Machine Learning-based system to automatically identify Indian languages from audio using advanced acoustic feature extraction and classification models.

🚀 **Live App:** https://language-detection-n9dzfxjanxvfckurbhhs5x.streamlit.app 

📂 **GitHub Repo:** https://github.com/npshivhare/Language-Detection  

🎥 **Demo Video:** 

---

## 📌 Overview

India is one of the most linguistically diverse countries with **22 official languages and 1000+ mother tongues**. This project focuses on **automatic spoken language identification** using classical machine learning models.

We perform a **comparative analysis of three models**:
- XGBoost (Best Performing)
- Random Forest
- Support Vector Machine (SVM)

The system classifies audio samples into **10 Indian languages** using acoustic features.

---

## 🎯 Key Features

- 🎧 Multi-class language classification from audio  
- 🧠 Feature extraction using MFCC, Chroma & Spectral Contrast  
- ⚖️ Comparative analysis of ML models  
- 📊 ROC curves & confusion matrix visualization  
- 🌐 Interactive Streamlit web application  
- 🔍 Confidence-based prediction system  

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Libraries:**  
  - Librosa (Audio Processing)  
  - NumPy, Pandas  
  - Scikit-learn  
  - XGBoost  
  - Matplotlib, Seaborn  
- **Deployment:** Streamlit  

---

## 📂 Dataset

- 📌 Source: Kaggle (Indian Languages Audio Dataset)  
- 🎧 Format: MP3 audio files  
- 🌍 Languages Covered:
  - Bengali, Gujarati, Hindi, Kannada  
  - Malayalam, Marathi, Punjabi  
  - Tamil, Telugu, Urdu  
- 🔊 Preprocessing:
  - Resampled to 16kHz  
  - Silence trimming  

---

## ⚙️ Methodology

### 🔹 Feature Extraction
Each audio file is converted into a **59-dimensional feature vector**:
- 40 MFCC coefficients  
- 12 Chroma features  
- 7 Spectral Contrast features  

### 🔹 Data Processing
- Label encoding  
- Feature scaling using `StandardScaler`  
- Train-test split (80/20)  

### 🔹 Models Used
- XGBoost (Gradient Boosting)
- Random Forest (Bagging Ensemble)
- SVM (RBF Kernel)

---

## 🤖 Model Performance

| Model           | Accuracy | AUC   | Performance |
|----------------|---------|-------|------------|
| XGBoost        | 96%     | 0.999 | ⭐ Best     |
| Random Forest  | ~60%    | High  | Moderate   |
| SVM            | ~82.7%  | High  | Good       |

✅ **XGBoost achieved the highest performance**

---

## 📊 Results & Insights

- XGBoost significantly outperforms other models on high-dimensional audio features
- MFCC + Chroma + Spectral Contrast provide strong discriminative power
- Some languages (e.g., Gujarati & Punjabi) show overlapping patterns  
- Ensemble methods perform better than standalone classifiers  

---

## 🖥️ Application Features

- Upload audio file (.mp3)  
- Get predicted language  
- View confidence scores from each model  
- Compare model outputs  
- Analyze performance visually  

---

## 🏆 Key Achievements

- Built a **complete audio ML pipeline** from scratch  
- Achieved **96% accuracy with XGBoost**  
- Performed **multi-model comparative analysis**  
- Worked with **real-world audio datasets**  
- Developed **deployable ML system using Streamlit**  

---

## 👨‍💻 Team Members

- Krish Naik  
- Nrependre Shivhare  
- Prasham Godha  

---

## 🙏 Mentors

- Dr. K. K. Sharma  
- Dr. Lalit Purohit  
- Dr. Upendra Singh  
- Mr. Akshay Gupta  

---

## 🔮 Future Work

- Extend to all **22 official Indian languages**  
- Use deep learning (CNN/RNN on spectrograms)  
- Real-time audio stream classification  
- Improve robustness with larger datasets  

---

## 📚 References

- Kaggle Dataset (Indian Languages Audio Dataset)  
- Librosa Documentation  
- Scikit-learn Documentation  
- Research papers on speech processing & ML  

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!
