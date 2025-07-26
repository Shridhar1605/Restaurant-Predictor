# 🍽️ Restaurant Visitor Predictor

This project predicts the number of visitors to a restaurant based on historical reservation and visit data using an **LSTM-based neural network**. It features a complete **end-to-end pipeline** with:

- A trained LSTM model
- A FastAPI backend
- A minimal HTML/CSS frontend

---

## 🚀 Tech Stack

- **Frontend**: HTML, CSS  
- **Backend**: FastAPI (Python)  
- **ML Model**: LSTM (Keras + TensorFlow)  
- **Data**: Historical air reservation data  

---

## 🧠 Features

- Predicts daily restaurant visitor counts
- Automatically scales inputs using `feature_scaler.pkl`
- Returns predicted value after inverse-scaling via `target_scaler.pkl`
- Modular and extensible for other time series applications

---

## 📁 Folder Structure
```
Restaurant_Predictor/
├── app/ # FastAPI backend
│ ├── main.py
│ ├── model_utils.py
├── data/ # Raw CSV and input test data
│ ├── air_reserve.csv
│ ├── sample_input.txt
├── models/ # Trained model + scalers
│ ├── lstm_model79.h5
│ ├── feature_scaler79.pkl
│ ├── target_scaler79.pkl
├── frontend/ # UI files
│ ├── index.html
│ └── style.css
├── README.md
```

---

## ⚙️ How to Run

### 🔧 Requirements

```bash
pip install fastapi uvicorn pandas scikit-learn tensorflow
```
---

### ▶️ Run the API
```bash
Copy code
cd app
uvicorn main:app --reload
```
### 🌐 Use the Frontend
Open frontend/index.html in a browser.
Paste the sample_input.txt data into the form and click Predict.
---
## 📈 Model Details
- Input: 21-day sliding window of historical features

- Output: Predicted visitor count

- LSTM units: 128

- R² Score: 79.9%

---

## 📌 To-Do / Improvements
- Add support for user-uploaded CSV

- Deploy API to a cloud platform (Render / Hugging Face Spaces)

- Add charts using JS or Streamlit

- Improve model accuracy with more features

