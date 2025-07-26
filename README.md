# Human Disease Prediction System Using Symptoms

## Overview

This project is a **machine learning-based application** that predicts diseases based on a user's selected symptoms. The system employs four different algorithms—**Decision Tree, Random Forest, K-Nearest Neighbour, and Naive Bayes**—to deliver diverse and reliable predictions. The intuitive user interface is now built with **Streamlit**, offering a seamless and interactive web experience. All prediction results and user data are stored in an **SQLite database** for easy reference and future analysis.

## 🌐 Live Demo
---
https://human-disease-prediction-system.streamlit.app/
---

## Features

- **Multiple Machine Learning Algorithms:** Utilizes Decision Tree, Random Forest, K-Nearest Neighbour, and Naive Bayes, providing multi-perspective predictions for enhanced reliability.
- **Modern Web Interface:** Built with Streamlit, enabling users to interactively select symptoms, view real-time predictions, and enjoy a responsive web-based experience.
- **Consolidated Predictions:** After running all algorithms, users can click the “Final Prediction” button for a consensus result, increasing confidence in the diagnosis.
- **Persistent Data Storage:** Patient information and predictions are stored in SQLite, supporting export and further analysis.
- **Easy-to-Use Workflow:** Streamlined process from symptom selection to result display, suitable for both technical and non-technical users.

## Tech Stack

| Component        | Technology           |
|------------------|---------------------|
| Programming      | Python              |
| GUI              | Streamlit           |
| ML Algorithms    | Decision Tree, Random Forest, K-Nearest Neighbour, Naive Bayes |
| Database         | SQLite              |
| Development Env. | Jupyter Notebook, VS Code |

## How It Works

1. **Input:**  
   The user enters their name and selects five symptoms from a predefined list.

2. **Prediction:**  
   Upon clicking “Predict All,” four machine learning models process the symptoms and predict potential diseases.

3. **Final Outcome:**  
   Clicking “Final Prediction” displays a consolidated disease prediction, synthesized from all model outputs.

4. **Data Storage:**  
   All patient details (name, symptoms, predictions) are recorded in an SQLite database.

## Model Performance

| Algorithm              | Accuracy |
|------------------------|----------|
| Decision Tree          | 93%      |
| Random Forest          | 95%      |
| K-Nearest Neighbour    | 92%      |
| Naive Bayes            | 93%      |

## Setup Instructions

1. Ensure Python is installed.
2. Install dependencies:  
```
pip install streamlit scikit-learn pandas numpy
```
3. Download the source code, as well as the training and testing datasets.
4. Open a terminal, navigate to the project folder, and launch the app:  
```
streamlit run app.py
```
5. Open the displayed URL in your browser to use the system.

## Future Improvements

- Add more symptoms and diseases to broaden model coverage.
- Incorporate more advanced ML algorithms (e.g., SVM, Neural Networks) for better accuracy.
- Extend data storage and analytics features for healthcare professionals.
- Integrate with health APIs for real-time updates.
- Add authentication and reporting functionality for medical use cases.

## Conclusion

By combining **multiple machine learning models** and a **modern Streamlit interface**, this project delivers a robust, user-friendly disease prediction tool. Its reliability, accessibility, and adaptability make it a valuable demonstration of applied AI in healthcare.

