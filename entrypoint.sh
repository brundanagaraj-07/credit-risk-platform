#!/usr/bin/env bash
set -e

echo "=============================================="
echo " NeoStats Credit Risk Intelligence Platform"
echo "=============================================="

if [ ! -f "/app/models/credit_risk_model.pkl" ]; then
    echo "[entrypoint] No trained model found -> running training pipeline..."
    echo "[entrypoint] (uses real /app/data/application_train.csv if present, otherwise auto-generates synthetic demo data)"
    python -m src.ml.train
else
    echo "[entrypoint] Existing trained model found in /app/models -> skipping training."
fi

echo "[entrypoint] Launching Streamlit UI on port 8501..."
exec streamlit run ui/app.py --server.address=0.0.0.0 --server.port=8501
