import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# ===============================
# Load Model & Scaler
# ===============================
model = joblib.load('machine_failure_model.pkl')
scaler = joblib.load('scaler.pkl')

# ===============================
# 1️⃣ Input: Device Data
# (يمكن استبدالها بقراءة من API أو Form لاحقًا)
# ===============================
data = {
    'equipment_id': 'EQ-01',
    'Air temperature [K]': 300,
    'Process temperature [K]': 310,
    'Rotational speed [rpm]': 1500,
    'Torque [Nm]': 40,
    'Tool wear [min]': 180,
    'Type_L': 0,
    'Type_M': 1
}

# ===============================
# 2️⃣ Prepare DataFrame
# ===============================
input_df = pd.DataFrame([data])

equipment_id = input_df['equipment_id'][0]
input_features = input_df.drop(columns=['equipment_id'])

# ===============================
# 3️⃣ Scaling
# ===============================
input_scaled = scaler.transform(input_features)

# ===============================
# 4️⃣ Prediction
# ===============================
pred_class = model.predict(input_scaled)[0]
pred_prob = model.predict_proba(input_scaled)[0][1]

failure_label = 'Failure' if pred_class == 1 else 'No Failure'

# ===============================
# 5️⃣ Estimate Days to Failure (Proxy)
# ===============================
# Using Tool Wear as simple Remaining Useful Life proxy
MAX_TOOL_WEAR = 250  # assumed threshold
remaining_days = max(0, int((MAX_TOOL_WEAR - data['Tool wear [min]']) * 0.5))

# ===============================
# 6️⃣ Build Output for Maintenance Tool
# ===============================
output = {
    'equipment_id': equipment_id,
    'predicted_failure_prob': round(pred_prob, 3),
    'days_to_failure': remaining_days,
    'last_maintenance': datetime.today().date()
}

output_df = pd.DataFrame([output])

# ===============================
# 7️⃣ Save CSV
# ===============================
output_df.to_csv('model_output.csv', index=False)

print("✅ Prediction completed")
print(output_df)
