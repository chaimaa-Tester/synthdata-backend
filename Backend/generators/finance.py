import pandas as pd

def generate_financeData(rows, rowCount):
    data = [{"Kunde": "Müller", "Einkommen": 4000, "Kredit": 20000} for _ in range(rowCount)]
    return pd.DataFrame(data)
