from fastapi import FastAPI, Request
from typing import List
from field_schemas import FieldDefinition, ExportRequest
import field_storage
from fastapi.middleware.cors import CORSMiddleware
from io import StringIO
from fastapi.responses import StreamingResponse
from preview_generator import generate_dummy_data


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


@app.post("/api/export")
async def export_csv(request: ExportRequest):
    # Dummy-Daten generieren
    df = generate_dummy_data(request.rows, request.rowCount)

    # CSV konfigurieren
    separator = "," if request.format.upper() == "CSV" else ";"
    line_end = "\r\n" if "CRLF" in request.lineEnding.upper() else "\n"

    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False, sep=separator, lineterminator=line_end)
    csv_buffer.seek(0)

    return StreamingResponse(
        csv_buffer,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=synthdata.csv"},
    )


# @app.post("/api/export")
# async def export_debug(request: Request):
#     body = await request.json()
#     print("📦 Eingehende Rohdaten vom Frontend:")
#     print(body)
#     return {"status": "debug"}



 

# 2. TO-DO CSV Export Funktion erstellen