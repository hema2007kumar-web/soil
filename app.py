from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model, label encoder, and the exact feature order used in training
model = joblib.load("model/soil_fertility_model.pkl")
le = joblib.load("model/label_encoder.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")

# These are the raw soil nutrient inputs the form will ask for.
# (Same as your dataset's input columns, before get_dummies was applied —
# for this dataset there are no categorical columns, so this list matches
# feature_columns directly. If your CSV has extra categorical columns,
# add their raw names here instead of the dummy-encoded ones.)
RAW_INPUT_COLUMNS = feature_columns


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Build a single-row DataFrame from the form, in the same
            # raw column order used before get_dummies was applied.
            input_data = {col: [float(request.form[col])] for col in RAW_INPUT_COLUMNS}
            input_df = pd.DataFrame(input_data)

            # Apply the same dummy-encoding step used in training.
            # (No-op here since this dataset is fully numeric, but this
            # keeps the app consistent with train_model.py if you add
            # categorical soil columns later.)
            input_df = pd.get_dummies(input_df, drop_first=True)

            # Make sure the input has exactly the columns the model expects,
            # in the right order — fill any missing dummy columns with 0.
            input_df = input_df.reindex(columns=feature_columns, fill_value=0)

            pred_encoded = model.predict(input_df)[0]
            prediction = str(le.inverse_transform([pred_encoded])[0])
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", feature_columns=RAW_INPUT_COLUMNS, prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
