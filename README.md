# 📰 Fake News Detector using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn)
![License](https://img.shields.io/badge/License-MIT-green)

An intelligent **Fake News Detection** web application built with **Python, Machine Learning, Natural Language Processing (NLP), Streamlit, and Scikit-learn**.

The application analyzes news headlines or articles and predicts whether they are **REAL** or **FAKE** using a TF-IDF based machine learning model. It also provides prediction confidence and explains which words contributed most to the decision.

---

## 📌 Features

✅ Detects whether a news article is **REAL** or **FAKE**

✅ Displays prediction confidence score

✅ Highlights important words influencing the prediction

✅ Interactive Streamlit web interface

✅ Automatic text preprocessing

✅ TF-IDF Vectorization

✅ Logistic Regression & Passive Aggressive Classifier

✅ Model evaluation with Accuracy, Precision, Recall and F1-Score

---

## 📸 Application Preview

### Home Page
![Home Page](https://github.com/user-attachments/assets/37a092bc-ce5c-46e9-8307-37d5311e1af5)

### Prediction Result
![Prediction Result](https://github.com/user-attachments/assets/526ca438-0687-4ea3-bb9e-8ca4e8f3f268)

---

# 🛠️ Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Joblib
- NLTK
- TF-IDF Vectorizer

---

# 📂 Project Structure

```text
fake-news-detector/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── generate_dataset.py
│   └── news_dataset.csv
│
├── models/
│   ├── model.joblib
│   ├── vectorizer.joblib
│   └── metadata.joblib
│
└── src/
    ├── preprocess.py
    └── train.py
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/nihahawas/fake-news-detector.git
```

Move into the project directory

```bash
cd fake-news-detector
```

Create a virtual environment

```bash
python -m venv venv
```

Activate it

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 📊 Generate Dataset

```bash
python data/generate_dataset.py
```

This creates a balanced synthetic dataset for demonstration purposes.

---

# 🤖 Train the Model

```bash
python src/train.py
```

The training pipeline will

- Clean text
- Remove stopwords
- Lemmatize words
- Generate TF-IDF features
- Train multiple ML models
- Compare model performance
- Save the best model automatically

---

# 🚀 Run the Application

```bash
streamlit run app.py
```

Open

```
http://localhost:8501
```

---

# 🧠 Machine Learning Pipeline

```
News Article
      │
      ▼
Text Cleaning
      │
      ▼
Tokenization
      │
      ▼
Stopword Removal
      │
      ▼
Lemmatization
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Machine Learning Model
      │
      ▼
Prediction
      │
      ▼
REAL / FAKE
```

---

# 📈 Model Evaluation

The model is evaluated using

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

# 🌟 Future Improvements

- Fine-tune BERT/DistilBERT
- Deploy using Docker
- REST API using FastAPI
- User Authentication
- Multi-language Fake News Detection
- Live News API Integration
- Model Monitoring with MLflow

---

# 👩‍💻 Author

## Niha Hawas

Computer Science Student | Machine Learning Enthusiast | AI Developer

**GitHub**

https://github.com/nihahawas

**LinkedIn**

https://www.linkedin.com/in/nihahawas45/

**Portfolio**

https://nihahawas.github.io/personal-portfolio-website/

---

# ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.

---

# 📄 License

This project is developed for educational and portfolio purposes.
