# ❤️ Heart Disease Risk XAI – Hybrid AI

A comprehensive Streamlit application that predicts heart disease risk using a hybrid AI approach. This project features calibrated machine learning models, SHAP (SHapley Additive exPlanations) for interpretability, and a "What-if" simulator for dynamic risk assessment.

## 🚀 Features

* **Hybrid AI Model:** Uses a calibrated pipeline to predict heart disease risk probabilities.
* **XAI (Explainable AI):** visualizes *why* a specific prediction was made using SHAP waterfall plots.
* **Interactive Simulation:** Adjust patient vitals (Cholesterol, BP, etc.) in real-time to see how lifestyle changes impact risk.
* **User-Friendly Interface:** Built with Streamlit for easy accessibility.

## 📂 Project Structure

* `app.py`: The main Streamlit application script.
* `model_loader.py`: Handles data loading, preprocessing, model training, and artifact management.
* `requirements.txt`: List of Python dependencies.

## 🛠️ Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/heart-disease-xai.git](https://github.com/YOUR_USERNAME/heart-disease-xai.git)
    cd heart-disease-xai
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app:**
    ```bash
    streamlit run app.py
    ```

## 📊 Data Source
The model is trained on the [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease) (or specify your specific source here).
