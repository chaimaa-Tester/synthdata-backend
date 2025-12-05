from fastapi import FastAPI, UploadFile, File, Form
from typing import List
from pydantic import BaseModel
from field_schemas import FieldDefinition, ExportRequest
import field_storage
from fastapi.middleware.cors import CORSMiddleware
import io
import pandas as pd
from fastapi.responses import StreamingResponse, JSONResponse
from generators.health import generate_healthData
from generators.finance import generate_financeData
from generators.container import generate_containerData
from dbhandler import connect_to_DB
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

    for ucid in usedUseCaseIds:
        if ucid.lower() == "logistik":
            df = generate_containerData(request.rows, request.rowCount)
        elif ucid.lower() == "gesundheit":
            df = generate_healthData(request.rows, request.rowCount)
        elif ucid.lower() == "finanzen":
            df = generate_financeData(request.rows, request.rowCount)
            
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



# @app.post("/api/export")
# async def export_debug(request: Request):
#     body = await request.json()
#     print("📦 Eingehende Rohdaten vom Frontend:")
#     print(body)
#     return {"status": "debug"}


# ==== Custom Distribution Fitting API ====
class DistributionFitRequest(BaseModel):
    points: list[float]


@app.post("/api/fit-distribution")
async def fit_distribution(request: DistributionFitRequest):
    points = np.array(request.points, dtype=float)
    # Remove NaN/infinite
    points = points[np.isfinite(points)]
    if len(points) < 2:
        return JSONResponse(status_code=400, content={"error": "Not enough points"})
    # Normalize to [0, 1]
    min_val = np.min(points)
    max_val = np.max(points)
    if max_val - min_val > 0:
        norm_points = (points - min_val) / (max_val - min_val)
    else:
        norm_points = np.zeros_like(points)

    # List of distributions to test
    candidates = {
        "norm": stats.norm,
        "lognorm": stats.lognorm,
        "expon": stats.expon,
        "gamma": stats.gamma,
        "beta": stats.beta,
        "uniform": stats.uniform,
    }
    results = {}
    best_fit = None
    best_p = -np.inf
    best_params = None
    for name, dist in candidates.items():
        try:
            params = dist.fit(norm_points)
            D, p = stats.kstest(norm_points, dist.name, args=params)
            results[name] = {
                "parameters": [float(x) for x in params],
                "p_value": float(p),
            }
            if p > best_p:
                best_p = p
                best_fit = name
                best_params = [float(x) for x in params]
        except Exception:
            continue

    x_fit = np.linspace(0, 1, 100)
    y_fit = []
    if best_fit is not None and best_params is not None:
        dist = candidates[best_fit]
        try:
            y_fit = dist.pdf(x_fit, *best_params)
        except Exception:
            y_fit = []

    return {
        "best_distribution": best_fit,
        "p_value": float(best_p),
        "parameters": best_params if best_params is not None else [],
        "all_results": results,
        "fit_curve": {
            "x": x_fit.tolist(),
            "y": y_fit.tolist() if hasattr(y_fit, "tolist") else [],
        },
    }

# ==== Profile Management (JSON-Speicherung) ====
from storage_manager import load_profiles, add_profile, delete_profile, save_profile_data, get_profile_data

@app.get("/profiles")
def get_profiles():
    """Lädt alle gespeicherten Profile aus DB"""
    if not connect_to_DB():
        return JSONResponse(status_code=503, content={"error": "DB nicht verbunden"})
    
    return load_profiles(connect_to_DB())

@app.post("/profiles")
def create_profile(profile: dict):
    """Erstellt ein neues Profil und speichert es in DB"""
    name = profile.get("name")
    if not name:
        return {"error": "Name ist erforderlich"}
    if not connect_to_DB():
        return JSONResponse(status_code=503, content={"error": "DB nicht verbunden"})
    
    new_profile = add_profile(connect_to_DB(), name)
    return new_profile

@app.delete("/profiles/{db_id}")
def remove_profile(db_id: str):
    """Löscht ein bestehendes Profil aus data.json"""
    if not connect_to_DB():
        return JSONResponse(status_code=503, content={"error": "DB nicht verbunden"})
    
    delete_profile(connect_to_DB(), db_id)
    return {"message": f"Profil {db_id} wurde gelöscht"}


# ==== Profile Data Storage ====
@app.post("/profiles/{db_id}/data")
def save_profile_data_route(db_id: str, data: dict):
    """Speichert Daten innerhalb eines bestimmten Profils"""
    if not connect_to_DB():
        return JSONResponse(status_code=503, content={"error": "DB nicht verbunden"})
    
    try:
        save_profile_data(connect_to_DB(), db_id, data)
        return {"message": f"Daten für Profil {db_id} wurden gespeichert"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/profiles/{db_id}/data")
def get_profile_data_route(db_id: str):
    """Lädt gespeicherte Daten für ein bestimmtes Profil"""
    if not connect_to_DB():
        return JSONResponse(status_code=503, content={"error": "DB nicht verbunden"})
    
    try:
        data = get_profile_data(connect_to_DB(), db_id)
        if data is None:
            return {"data": {}}
        return {"data": data}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})