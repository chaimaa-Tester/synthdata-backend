from fastapi import FastAPI
from typing import List
from field_schemas import FieldDefinition
import field_storage


app = FastAPI()

@app.post("/api/v1/fields")
async def receive_fields(fields: List[FieldDefinition]):
    field_storage.save_fields(fields)
    return {"status": "ok", "received_fields": len(fields)}


@app.get("/api/v1/fields")
async def list_fields():
    return field_storage.get_fields()