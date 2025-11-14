from fastapi import FastAPI, Request, UploadFile, File, Form
from typing import List
from pydantic import BaseModel
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
    """Lädt alle gespeicherten Profile aus data.json"""
    return load_profiles()

@app.post("/profiles")
def create_profile(profile: dict):
    """Erstellt ein neues Profil und speichert es in data.json"""
    name = profile.get("name")
    if not name:
        return {"error": "Name ist erforderlich"}
    new_profile = add_profile(name)
    return new_profile

@app.delete("/profiles/{profile_id}")
def remove_profile(profile_id: str):
    """Löscht ein bestehendes Profil aus data.json"""
    delete_profile(profile_id)
    return {"message": f"Profil {profile_id} wurde gelöscht"}


# ==== Profile Data Storage ====
@app.post("/profiles/{profile_id}/data")
def save_profile_data_route(profile_id: str, data: dict):
    """Speichert Daten innerhalb eines bestimmten Profils"""
    try:
        save_profile_data(profile_id, data)
        return {"message": f"Daten für Profil {profile_id} wurden gespeichert"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/profiles/{profile_id}/data")
def get_profile_data_route(profile_id: str):
    """Lädt gespeicherte Daten für ein bestimmtes Profil"""
    try:
        data = get_profile_data(profile_id)
        if data is None:
            return {"data": {}}
        return {"data": data}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})