from flask import Flask, render_template, request
import pickle
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("Model.pkl", "rb"))
scaler = pickle.load(open("standar_scaler.pkl", "rb"))

# 🔥 IMPORTANT: trained column order
trained_columns = [
    'MonthlyCharges_qt', 'TotalChargesreplaced_qt', 'tenure_trim', 'gender',
    'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
    'SIM_provider', 'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
    'Contract_odinal'
]


@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        try:
            # =========================
            # Get Inputs
            # =========================
            data = {
                "MonthlyCharges_qt": float(request.form['MonthlyCharges']),
                "TotalChargesreplaced_qt": float(request.form['TotalCharges']),
                "tenure_trim": int(request.form['tenure']),
                "gender": int(request.form['gender']),
                "Partner": int(request.form['Partner']),
                "Dependents": int(request.form['Dependents']),
                "PhoneService": int(request.form['PhoneService']),
                "PaperlessBilling": int(request.form['PaperlessBilling']),
                "SIM_provider": request.form['SIM_provider'],
                "MultipleLines": request.form['MultipleLines'],
                "InternetService": request.form['InternetService'],
                "OnlineSecurity": request.form['OnlineSecurity'],
                "OnlineBackup": request.form['OnlineBackup'],
                "DeviceProtection": request.form['DeviceProtection'],
                "TechSupport": request.form['TechSupport'],
                "StreamingTV": request.form['StreamingTV'],
                "StreamingMovies": request.form['StreamingMovies'],
                "Contract_odinal": request.form['Contract']
            }

            # =========================
            # Convert to DataFrame
            # =========================
            df = pd.DataFrame([data])

            # =========================
            # One-hot encoding
            # =========================
            df = pd.get_dummies(df)

            # =========================
            # Match trained columns
            # =========================
            df = df.reindex(columns=trained_columns, fill_value=0)

            # =========================
            # Scale
            # =========================
            scaled = scaler.transform(df)

            # =========================
            # Predict
            # =========================
            pred = model.predict(scaled)[0]

            if pred == 0:
                result = "✅ Customer will Stay"
            else:
                result = "⚠️ Customer will Churn"

        except Exception as e:
            result = f"Error: {e}"

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)