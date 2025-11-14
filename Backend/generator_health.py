from typing import List
import pandas as pd
from preview_generator import generate_dummy_data
from field_schemas import FieldDefinition

def generate_healthData(rows: List[FieldDefinition], rowCount: int) -> pd.DataFrame:
    """
    Erzeugt Datensatz für Use Case 'gesundheit'.
    rows: Liste der Felddefinitionen vom Frontend (wie main.py übergeben).
    rowCount: Anzahl Zeilen.
    """
    # Falls generate_dummy_data direkt mit Frontend-Fields arbeitet:
    df = generate_dummy_data(rows, rowCount, as_text_for_sheets=False)
    return df