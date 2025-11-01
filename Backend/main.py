from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from typing import List
from field_schemas import FieldDefinition, ExportRequest
import field_storage
from fastapi.middleware.cors import CORSMiddleware
import io
import pandas as pd
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from generators.health import generate_healthData
from generators.finance import generate_financeData
from generators.logistic import generate_logisticData
from scipy import stats
import numpy as np

app = FastAPI()

# === CORS-Konfiguration ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Backend läuft"}

# === Felder speichern / abrufen ===
@app.post("/api/v1/fields")
async def receive_fields(fields: List[FieldDefinition]):
    field_storage.save_fields(fields)
    return {"status": "ok", "received_fields": len(fields)}

@app.get("/api/v1/fields")
async def list_fields():
    return field_storage.get_fields()


# === Export-Endpunkt ===
@app.post("/api/export")
async def export_data(request: ExportRequest):
    print(request)
    usedUseCaseIds = request.usedUseCaseIds
    df_list = []

    for ucid in usedUseCaseIds:
        if ucid.lower() == "containerlogistik":
            df_list.append(generate_logisticData(request.rows, request.rowCount))
        elif ucid.lower() == "gesundheit":
            df_list.append(generate_healthData(request.rows, request.rowCount))
        elif ucid.lower() == "finanzen":
            df_list.append(generate_financeData(request.rows, request.rowCount))

    if not df_list:
        raise HTTPException(status_code=400, detail="Keine Daten für die angegebenen UseCases gefunden")

    df = pd.concat(df_list, ignore_index=True)
    fmt = request.format.upper()

    # === JSON Export ===
    if fmt == "JSON":
        json_str = df.to_json(orient="records", force_ascii=False, indent=2)
        json_buffer = io.BytesIO(json_str.encode("utf-8"))
        json_buffer.seek(0)
        return StreamingResponse(
            json_buffer,
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=synthdata.json"},
        )

    # === SQL Export ===
    elif fmt == "SQL":
        table_name = getattr(request, "tableName", "exported_table")
        sql_buffer = io.StringIO()

        for _, row in df.iterrows():
            columns = ", ".join(row.index)
            values = []
            for v in row.values:
                if pd.isna(v):
                    values.append("NULL")
                else:
                    safe_val = str(v).replace("'", "''")
                    values.append(f"'{safe_val}'")
            sql_buffer.write(f"INSERT INTO {table_name} ({columns}) VALUES ({', '.join(values)});\n")

        sql_buffer.seek(0)
        return StreamingResponse(
            sql_buffer,
            media_type="application/sql",
            headers={"Content-Disposition": "attachment; filename=synthdata.sql"},
        )

    # === XLSX Export ===
    elif fmt == "XLSX":
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Sheet 1")
        output.seek(0)
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=synthdatawizard.xlsx"},
        )

    # === CSV Export (Standard) ===
    else:
        separator = "," if fmt == "CSV" else ";"
        line_end = "\r\n" if "CRLF" in request.lineEnding.upper() else "\n"

        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, sep=separator, lineterminator=line_end)
        csv_buffer.seek(0)
        return StreamingResponse(
            csv_buffer,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=synthdata.csv"},
        )


# === Distribution Detection ===
@app.post("/detect-distribution")
async def detect_distribution(file: UploadFile = File(...)):
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
async def detect_distribution_column(file: UploadFile = File(...), column: str = Form(...)):
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
            params = dist.fit(data)
            D, p = stats.kstest(data, dist.name, args=params)
            results[name] = {"parameters": params, "p_value": p}
            if p > best_p:
                best_p = p
                best_fit = name
        except Exception:
            continue

    hist_counts, bin_edges = np.histogram(data, bins=10, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    dist_curves = {}
    for name, dist in distributions.items():
        if name in results:
            params = results[name]["parameters"]
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
