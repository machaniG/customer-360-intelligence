import pandas as pd
import numpy as np
import datetime


def build_churn_labels(
    df: pd.DataFrame,
    as_of_date: str,
    horizon_months: int = 1
) -> pd.DataFrame:

    """
    Monthly churn labeling.

    Customers are labeled as:
        0 = active (purchase in next calendar month)
        1 = churned (no purchase in next calendar month)
        as_of_date = 2010-09-30
        label window = 2010-10-01 → 2010-10-31
    """

    as_of_date = pd.to_datetime(as_of_date)

    # Start of next month
    next_month_start = (as_of_date + pd.offsets.MonthBegin(1))

    # End of next month
    next_month_end = (
        next_month_start + pd.offsets.MonthEnd(horizon_months)
    )

    # Safety check (prevents 100% churn bug)
    max_date = pd.to_datetime(df["InvoiceDate"]).max()

    if next_month_end > max_date:
        raise ValueError(
            f"Cannot build labels. "
            f"Label window ends at {next_month_end.date()}, "
            f"but dataset ends at {max_date.date()}."
        )

    # Activity in label window
    future = df[
        (pd.to_datetime(df["InvoiceDate"]) >= next_month_start) &
        (pd.to_datetime(df["InvoiceDate"]) <= next_month_end)
    ]

    active_customers = set(future["Customer ID"].dropna().unique())

    customers = pd.DataFrame({
        "Customer ID": df["Customer ID"].dropna().unique()
    })

    customers["churn_label"] = customers["Customer ID"].apply(
        lambda cid: 0 if cid in active_customers else 1
    )

    return customers
