import pandas as pd

def generate_healthData(rows, rowCount):
    data = [{"Name": "Max", "Alter": 30, "Blutdruck": 120} for _ in range(rowCount)]
    return pd.DataFrame(data)
