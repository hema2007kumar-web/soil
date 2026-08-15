# Soil Fertility Predictor

A machine learning web app that predicts soil fertility level (Less Fertile / Fertile / Highly Fertile) based on soil nutrient test values, built with a RandomForest classifier and a Flask frontend.

## Overview

This project takes standard soil test parameters — nitrogen, phosphorus, potassium, pH, electrical conductivity, organic carbon, and key micronutrients — and predicts the overall fertility class of the soil sample. It's meant to help simplify soil health assessment for agricultural use.

## Dataset

Trained on the [Soil Fertility Dataset](https://www.kaggle.com/datasets/rahuljaiswalonkaggle/soil-fertility-dataset) from Kaggle (880 soil samples, 12 physicochemical features, 1 fertility label).

**Features used:**
| Feature | Description |
|---|---|
| N | Nitrogen |
| P | Phosphorus |
| K | Potassium |
| ph | Soil pH |
| ec | Electrical conductivity |
| oc | Organic carbon |
| S | Sulfur |
| zn | Zinc |
| fe | Iron |
| cu | Copper |
| Mn | Manganese |
| B | Boron |

**Target (`Output`):** 0 = Less Fertile, 1 = Fertile, 2 = Highly Fertile

## Tech stack

- **Model:** scikit-learn `RandomForestClassifier`
- **Preprocessing:** `LabelEncoder` for the target, `pandas.get_dummies` for any categorical inputs
- **Backend:** Flask
- **Frontend:** HTML form (Jinja2 templates)
- **Model persistence:** joblib
- **Deployment:** Render (via `gunicorn`)

## Project structure

```
├── app.py                  # Flask web app
├── train_model.py          # Model training script
├── requirements.txt        # Python dependencies
├── Procfile                 # Deployment start command
├── dataset1.csv             # Training dataset
├── model/                   # Saved model, encoder, and feature list (generated)
│   ├── soil_fertility_model.pkl
│   ├── label_encoder.pkl
│   └── feature_columns.pkl
└── templates/
    └── index.html            # Prediction form UI
```

## Running locally

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Train the model (generates the `model/` folder):
   ```
   python train_model.py
   ```
3. Run the app:
   ```
   python app.py
   ```
4. Open `http://127.0.0.1:5000` in your browser.

## How it works

1. The user enters soil test values into the web form.
2. The Flask backend loads the trained RandomForest model, applies the same preprocessing used during training, and predicts a fertility class.
3. The predicted class is decoded back to its original label and shown to the user.

## Future improvements

- Add crop recommendations based on predicted fertility level
- Support batch predictions via CSV upload
- Add data visualizations for nutrient levels vs. fertility class
