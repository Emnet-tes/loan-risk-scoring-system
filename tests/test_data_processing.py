import pandas as pd
from datetime import datetime, timedelta

def test_calculate_rfm_basic_case():
    snapshot = datetime(2024, 1, 1)
    df = pd.DataFrame({
        'CustomerId': ['A', 'A', 'B'],
        'TransactionStartTime': [
            snapshot - timedelta(days=10),
            snapshot - timedelta(days=5),
            snapshot - timedelta(days=20)
        ],
        'TransactionId': [1, 2, 3],
        'Value': [100, 150, 80]
    })

    rfm = calculate_rfm(df, snapshot)
    assert 'Recency' in rfm.columns
    assert 'Frequency' in rfm.columns
    assert 'Monetary' in rfm.columns
    assert rfm.loc[rfm['CustomerId'] == 'A', 'Frequency'].values[0] == 2

def calculate_rfm(df, snapshot_date):
    return df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
        'TransactionId': 'count',
        'Value': 'sum'
    }).reset_index().rename(columns={
        'TransactionStartTime': 'Recency',
        'TransactionId': 'Frequency',
        'Value': 'Monetary'
    })