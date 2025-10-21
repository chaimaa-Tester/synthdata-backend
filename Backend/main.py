from fastapi import FastAPI, Request, UploadFile, File, Form
from typing import List
from field_schemas import FieldDefinition, ExportRequest
import field_storage
from fastapi.middleware.cors import CORSMiddleware
import io
import pandas as pd
from fastapi.responses import StreamingResponse, JSONResponse
from preview_generator import generate_dummy_data
from scipy import stats
import numpy as np

app = FastAPI()
@app.get("/")
def root():
    return {"message": "Backend läuft"}


# CORS-Konfiguration (z. B. für Frontend auf Port 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # oder ["*"] für Entwicklung
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


@app.post("/api/export")
async def export_csv(request: ExportRequest):
    print(request)
    # Dummy-Daten generieren
    df = generate_dummy_data(request.rows, request.rowCount)

    if request.format.upper() == "XLSX":
        # Mehrere Blätter unterstützen
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            # Beispiel: Ein Tabellenblatt "Sheet 1"
            df.to_excel(writer, index=False, sheet_name="Sheet 1")
        output.seek(0)
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=synthdatawizard.xlsx"},
        )
    elif request.format.upper() == "JSON":
        # JSON (pretty, records)
        json_text = df.to_json(orient="records", force_ascii=False, indent=2)
        json_buffer = io.StringIO(json_text)
        json_buffer.seek(0)
        return StreamingResponse(
            json_buffer,
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=synthdata.json"},
        )
    
    elif request.format.upper() == "SQL":
        # SQL: einfache INSERT INTO Statements
        table_name = getattr(request, "tableName", None) or getattr(request, "table_name", None) or "synthdata"

        def format_value(v):
            if pd.isna(v):
                return "NULL"
            if isinstance(v, bool):
                return "1" if v else "0"
            if isinstance(v, (int,)):
                return str(v)
            if isinstance(v, float):
                return str(v)
            s = str(v).replace("'", "''")
            return f"'{s}'"

        cols = ", ".join([f'"{c}"' for c in df.columns])
        statements = []
        for _, row in df.iterrows():
            vals = ", ".join(format_value(row[c]) for c in df.columns)
            statements.append(f"INSERT INTO \"{table_name}\" ({cols}) VALUES ({vals});")

        sql_text = "\n".join(statements)
        sql_buffer = io.StringIO(sql_text)
        sql_buffer.seek(0)
        return StreamingResponse(
            sql_buffer,
            media_type="application/sql",
            headers={"Content-Disposition": f"attachment; filename={table_name}.sql"},
        )
    else:
        # CSV konfigurieren
        separator = "," if request.format.upper() == "CSV" else ";"
        line_end = "\r\n" if "CRLF" in request.lineEnding.upper() else "\n"

        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, sep=separator, lineterminator=line_end)
        csv_buffer.seek(0)

        return StreamingResponse(
            csv_buffer,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=synthdata.csv"},
        )
         


@app.post("/detect-distribution")
async def detect_distribution(file: UploadFile = File(...)):
    # Datei einlesen (CSV oder XLSX)
    contents = await file.read()
    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            return JSONResponse(status_code=400, content={"error": "Unsupported file type"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Failed to read file: {str(e)}"})

    columns = df.columns.tolist()
    return {"columns": columns}


@app.post("/detect-distribution/column")
async def detect_distribution_column(
    file: UploadFile = File(...),
    column: str = Form(...)
):
    # Datei einlesen (CSV oder XLSX)
    contents = await file.read()
    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            return JSONResponse(status_code=400, content={"error": "Unsupported file type"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Failed to read file: {str(e)}"})

    if column not in df.columns:
        return JSONResponse(status_code=400, content={"error": "Column not found in data"})

    data = df[column].dropna().values
    # Nur numerische Daten verwenden
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return JSONResponse(status_code=400, content={"error": "Selected column has no valid numerical data"})

    distributions = {
        "norm": stats.norm,
        "expon": stats.expon,
        "gamma": stats.gamma,
        "lognorm": stats.lognorm,
        "uniform": stats.uniform,
    }

    best_fit = None
    best_p = -np.inf
    results = {}

    for name, dist in distributions.items():
        try:
            # Fit distribution to data
            params = dist.fit(data)
            # KS-Test
            D, p = stats.kstest(data, dist.name, args=params)
            results[name] = {
                "parameters": params,
                "p_value": p,
            }
            if p > best_p:
                best_p = p
                best_fit = name
        except Exception:
            # Falls Fit oder Test fehlschlägt, ignoriere
            continue

    # Histogramm-Werte (10 Bins)
    hist_counts, bin_edges = np.histogram(data, bins=10, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Verteilungskurven für alle Distributionen berechnen
    dist_curves = {}
    for name, dist in distributions.items():
        if name in results:
            params = results[name]["parameters"]
            # pdf für bin_centers berechnen
            try:
                pdf_vals = dist.pdf(bin_centers, *params)
                dist_curves[name] = pdf_vals.tolist()
            except Exception:
                dist_curves[name] = []

    return {
        "best_distribution": best_fit,
        "p_value": best_p,
        "parameters": results.get(best_fit, {}).get("parameters", []),
        "values": data.tolist(),
        "histogram": {
            "counts": hist_counts.tolist(),
            "bin_edges": bin_edges.tolist(),
            "bin_centers": bin_centers.tolist(),
        },
        "distribution_curves": dist_curves,
    }



# @app.post("/api/export")
# async def export_debug(request: Request):
#     body = await request.json()
#     print("📦 Eingehende Rohdaten vom Frontend:")
#     print(body)
#     return {"status": "debug"}