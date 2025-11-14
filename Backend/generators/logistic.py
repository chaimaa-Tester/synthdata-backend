import pandas as pd

def generate_logisticData(rows, rowCount):
    data = [{"Container": "C123", "Status": "Versendet", "Ort": "Hamburg"} for _ in range(rowCount)]
    return pd.DataFrame(data)
