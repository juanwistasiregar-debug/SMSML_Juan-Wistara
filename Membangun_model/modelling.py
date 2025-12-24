import pandas as pd
import mlflow
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Inisialisasi DagsHub
USERNAME = "juanwistasiregar"
REPO_NAME = "Eksperimen_SML_Juan-Wistara"
dagshub.init(repo_owner=USERNAME, repo_name=REPO_NAME, mlflow=True)
mlflow.set_tracking_uri(f"https://dagshub.com/{USERNAME}/{REPO_NAME}.mlflow")

# 2. Aktifkan Autolog (Syarat Kriteria Basic)
# Ini akan mencatat semua parameter & metrik secara otomatis
mlflow.sklearn.autolog()

# 3. Load Data
df = pd.read_csv('Membangun_model/churn_preprocessing.csv')
X = df.drop(columns=['Churn'])
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Pelatihan Model (Tanpa manual logging yang panjang)
with mlflow.start_run(run_name="RF_Basic_Juan_Wistara"):
    # Model standar tanpa tuning berlebihan
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc}")

print("✅ Berhasil! Modelling Basic (Autolog) selesai.")