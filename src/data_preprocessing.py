import os
import pandas as pd
import numpy as np


def create_dataset():
    np.random.seed(42)

    n = 1000

    data = {
        "customer_id": [f"CUST{i:04d}" for i in range(1, n + 1)],
        "gender": np.random.choice(["Male", "Female"], n),
        "age": np.random.randint(18, 70, n),
        "tenure_months": np.random.randint(1, 73, n),
        "monthly_charges": np.round(np.random.uniform(20, 120, n), 2),
        "contract_type": np.random.choice(
            ["Month-to-month", "One year", "Two year"],
            n,
            p=[0.55, 0.25, 0.20]
        ),
        "payment_method": np.random.choice(
            ["Credit Card", "Bank Transfer", "Electronic Check"], n
        ),
        "internet_service": np.random.choice(
            ["DSL", "Fiber optic", "No"], n
        ),
        "support_calls": np.random.randint(0, 10, n),
        "streaming_service": np.random.choice(["Yes", "No"], n),
        "senior_citizen": np.random.choice([0, 1], n, p=[0.85, 0.15]),
        "paperless_billing": np.random.choice(["Yes", "No"], n)
    }

    df = pd.DataFrame(data)

    df["total_charges"] = np.round(
        df["monthly_charges"] * df["tenure_months"], 2
    )

    churn_score = (
        0.15
        + 0.25 * (df["contract_type"] == "Month-to-month")
        + 0.15 * (df["internet_service"] == "Fiber optic")
        + 0.05 * (df["support_calls"] >= 5)
        + 0.10 * (df["monthly_charges"] > 80)
        - 0.20 * (df["tenure_months"] > 36)
        - 0.10 * (df["contract_type"] == "Two year")
    )

    churn_probability = np.clip(churn_score, 0.05, 0.90)

    df["churn"] = np.random.binomial(1, churn_probability)
    df["churn"] = df["churn"].map({1: "Yes", 0: "No"})

    return df


if __name__ == "__main__":
    df = create_dataset()
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/customer_churn.csv", index=False)

    print("Dataset created successfully!")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print("\nFirst 5 rows:")
    print(df.head())
