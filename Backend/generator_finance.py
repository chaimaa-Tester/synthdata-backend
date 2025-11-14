from typing import List
import pandas as pd
from preview_generator import generate_dummy_data
from field_schemas import FieldDefinition

def generate_financeData(rows: List[FieldDefinition], rowCount: int) -> pd.DataFrame:
    # Spezifische Vorbelegung / Regeln für Finanzdaten können hier ergänzt werden
    df = generate_dummy_data(rows, rowCount, as_text_for_sheets=False)
    return df