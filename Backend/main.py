from fastapi import FastAPI, Request
from typing import List
from field_schemas import FieldDefinition
import field_storage
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # oder ["*"] für Entwicklung
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/v1/fields")
async def receive_fields(fields: List[FieldDefinition]):
    field_storage.save_fields(fields)
    return {"status": "ok", "received_fields": len(fields)}

@app.get("/api/v1/fields")
async def list_fields():
    return field_storage.get_fields()


@app.post("/api/my-endpoint")
async def receive_data(request: Request):
    data = await request.json() # 1. TO-DO: Json-Daten empfangen und in Variablen speichern. (in field_schemas)
    print("Empfangene Daten:", data)
    return {"status": "ok", "message": "Daten empfangen!"}



# 2. TO-DO CSV Export Funktion erstellen