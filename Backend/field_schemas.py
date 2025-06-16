from pydantic import BaseModel
from typing import Optional, List

# Modell zur Beschreibung eines Feldes
class FieldDefinition(BaseModel):
    name: str
    type: str
    dependency: Optional[str] = None
    showinTable: Optional[bool] = True

# Speicher für empfangene JSON-Daten
stored_data: List[FieldDefinition] = []

# Funktion zum Speichern
def store_json_data(fields: List[FieldDefinition]):
    global stored_data
    stored_data = fields

# Funktion zum Abrufen
def get_stored_json_data() -> List[FieldDefinition]:
    return stored_data