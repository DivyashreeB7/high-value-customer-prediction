
import os
import io
import zipfile
import requests
import numpy as np
import pandas as pd
import gradio as gr

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

FEATURES = [
    "Recency", "Frequency", "Monetary",
    "AverageOrderValue", "ProductVariety",
    "ActiveSpanDays", "PurchaseRate"
]

def load_data():
    url = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        excel = next(
            n for n in z.namelist()
            if n.lower().endswith((".xlsx", ".xls"))
        )
        with z.open(excel) as f:
            return pd.read_excel(f)

def prepare_data(df):
    required = [
        "InvoiceNo", "StockCode", "Quantity",
        "InvoiceDate", "UnitPrice", "CustomerID"
    ]

    df = df.drop_duplicates().dropna(subset=required).copy()
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df = df[~df["InvoiceNo"].str.startswith("C")]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df = df[
        (df["Quantity"] > 0) &
        (df["UnitPrice"] > 0)
    ].copy()

    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["CustomerID"] = df["CustomerID"].astype(str)

    last_date = df["InvoiceDate"].max().normalize()
    cutoff = last_date - pd.Timedelta(days=90)

    history = df[df["InvoiceDate"] < cutoff].copy()
    future = df[df["InvoiceDate"] >= cutoff].copy()

    reference = history["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)

    customers = history.groupby("CustomerID").agg(
        LastPurchase=("InvoiceDate", "max"),
        FirstPurchase=("InvoiceDate", "min"),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum"),
        ProductVariety=("StockCode", "nunique")
    )

    customers["Recency"] = (
        reference - customers["LastPurchase"].dt.normalize()
    ).dt.days

    customers["ActiveSpanDays"] = (
        customers["LastPurchase"].dt.normalize()
        - customers["FirstPurchase"].dt.normalize()
    ).dt.days

    customers["AverageOrderValue"] = (
        customers["Monetary"] / customers["Frequency"].clip(lower=1)
    )

    customers["PurchaseRate"] = (
        customers["Frequency"]
        / (customers["ActiveSpanDays"].clip(lower=1) / 30)
    )

    future_spend = future.groupby("CustomerID")["Revenue"].sum()
    customers["FutureSpend"] = future_spend.reindex(
        customers.index, fill_value=0
    )

    positive = customers.loc[
        customers["FutureSpend"] > 0, "FutureSpend"
    ]

    if positive.empty:
        raise ValueError("No positive future spending found.")

    threshold = positive.quantile(0.80)

    customers["VIP"] = (
        (customers["FutureSpend"] > 0) &
        (customers["FutureSpend"] >= threshold)
    ).astype(int)

    data = customers[FEATURES + ["VIP"]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna(subset=["VIP"])

    if data["VIP"].nunique() < 2 or data["VIP"].value_counts().min() < 2:
        raise ValueError("Not enough examples in both VIP classes.")

    return data

def train_models():
    data = prepare_data(load_data())
    X = data[FEATURES]
    y = data["VIP"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                class_weight="balanced", max_iter=1000
            ))
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ))
        ])
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "model": model,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1-score": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, prob)
        }

    best_name = max(results, key=lambda n: results[n]["F1-score"])
    return results, best_name

# Train models when the hosted app starts.
RESULTS, BEST_NAME = train_models()
BEST_MODEL = RESULTS[BEST_NAME]["model"]

def predict_customer(
    recency, frequency, monetary, average_order_value,
    product_variety, active_span_days, purchase_rate
):
    values = [
        recency, frequency, monetary, average_order_value,
        product_variety, active_span_days, purchase_rate
    ]

    row = pd.DataFrame([values], columns=FEATURES)
    probability = float(BEST_MODEL.predict_proba(row)[0, 1])
    is_vip = probability >= 0.5

    prediction = "VIP / High-Value Customer" if is_vip else "Non-VIP Customer"

    if is_vip:
        suggestion = "Consider loyalty rewards, personalized offers, or priority service."
    elif probability >= 0.35:
        suggestion = "Consider an engagement offer to encourage repeat purchases."
    else:
        suggestion = "Continue regular engagement and monitor future purchases."

    return prediction, f"{probability:.1%}", suggestion

def show_metrics():
    table = pd.DataFrame(RESULTS).T.drop(columns=["model"])
    table.index.name = "Model"
    return table.reset_index().round(4)

with gr.Blocks(title="High-Value Customer Prediction") as demo:
    gr.Markdown(
        "# High-Value Customer Prediction\n"
        "Predict whether a customer may become a VIP using historical "
        "purchasing behaviour."
    )

    gr.Markdown(
        "Enter customer-level features. These values should be calculated "
        "using the same feature definitions as the training dataset."
    )

    with gr.Row():
        recency = gr.Number(label="Recency (days)", value=30)
        frequency = gr.Number(label="Frequency (invoices)", value=5)
        monetary = gr.Number(label="Monetary value", value=500)

    with gr.Row():
        aov = gr.Number(label="Average order value", value=100)
        variety = gr.Number(label="Product variety", value=5)
        active_span = gr.Number(label="Active span (days)", value=60)

    purchase_rate = gr.Number(label="Purchase rate", value=2.5)

    button = gr.Button("Predict Customer Value", variant="primary")
    prediction = gr.Textbox(label="Prediction")
    probability = gr.Textbox(label="VIP Probability")
    suggestion = gr.Textbox(label="Suggested Treatment")

    button.click(
        predict_customer,
        inputs=[
            recency, frequency, monetary, aov,
            variety, active_span, purchase_rate
        ],
        outputs=[prediction, probability, suggestion]
    )

    gr.Markdown("## Model Evaluation")
    gr.Markdown(f"**Selected model by test F1-score:** {BEST_NAME}")
    metrics = gr.Dataframe(value=show_metrics, label="Test-set metrics")
    refresh = gr.Button("Show Evaluation Metrics")
    refresh.click(show_metrics, outputs=metrics)

    gr.Markdown(
        "VIP is defined in this project as positive future spending "
        "at or above the 80th percentile during the final 90-day period. "
        "Predictions are estimates, not guarantees."
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)
