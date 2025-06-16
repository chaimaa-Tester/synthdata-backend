from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from field_schemas import FieldDefinition, store_json_data, get_stored_json_data
import field_storage  # bleibt für /api/v1/fields

app = FastAPI()

# CORS-Konfiguration (z. B. für Frontend auf Port 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # oder ["*"] für Entwicklung
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# /api/v1/fields bleibt unverändert
@app.post("/api/v1/fields")
async def receive_fields(fields: List[FieldDefinition]):
    field_storage.save_fields(fields)
    return {"status": "ok", "received_fields": len(fields)}

@app.get("/api/v1/fields")
async def list_fields():
    return field_storage.get_fields()

# Neuer Endpoint zum manuellen JSON-Empfang
@app.post("/api/my-endpoint")
async def receive_data(request: Request):
    data = await request.json()
    
    # Konvertiere JSON in FieldDefinition-Objekte
    if isinstance(data, list):
        fields = [FieldDefinition(**item) for item in data]
    else:
        fields = [FieldDefinition(**data)]

    # Speichern in field_schemas
    store_json_data(fields)

    for e in fields:
        print(e)

    print("Empfangene und gespeicherte Daten:", fields)
    return {"status": "ok", "message": "Daten empfangen und gespeichert!"}