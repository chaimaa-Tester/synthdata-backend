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
    x: list[float]
    y: list[float]


@app.post("/api/fit-distribution")
async def fit_distribution(request: DistributionFitRequest):
    from scipy.optimize import curve_fit

    # Convert lists to numpy arrays
    x = np.array(request.x, dtype=float)
    y = np.array(request.y, dtype=float)

    # Remove invalid values
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(x) < 5:
        return JSONResponse(status_code=400, content={"error": "Not enough points"})

    # Normalize x to [0, 1]
    x_min = x.min()
    x_max = x.max()
    denom = x_max - x_min
    if denom == 0:
        x = np.zeros_like(x)
    else:
        x = (x - x_min) / denom

    # Normalize y so that it integrates to 1 (approximate PDF)
    area = np.trapz(y, x)
    if area > 0:
        y = y / area

    def fit_dist(pdf, init_params):
        try:
            params, _ = curve_fit(pdf, x, y, p0=init_params, maxfev=5000)
            err = np.mean((y - pdf(x, *params)) ** 2)
            return params, err
        except Exception:
            return None, np.inf

    # Candidate distributions (parametric PDFs) - improved initial parameters
    candidates = {
        "gamma":   (lambda t, a, scale: stats.gamma.pdf(t, a, scale=scale), [3.0, 0.3]),
        "normal":  (lambda t, mu, sigma: stats.norm.pdf(t, mu, sigma),      [0.5, 0.15]),
        "lognorm": (lambda t, s, scale: stats.lognorm.pdf(t, s, scale=scale), [0.6, 0.4]),
        "expon":   (lambda t, scale: stats.expon.pdf(t, scale=scale),       [0.5]),
        "weibull": (lambda t, c, scale: stats.weibull_min.pdf(t, c, scale=scale), [2.0, 0.4]),
    }

    best_name = None
    best_err = np.inf
    best_params = None

    for name, (pdf, init) in candidates.items():
        params, err = fit_dist(pdf, init)
        if name == "expon":
            err *= 1.3
        if params is not None and err < best_err:
            best_name = name
            best_err = err
            best_params = params

    if best_name is None or best_params is None:
        return JSONResponse(status_code=400, content={"error": "Fitting failed"})

    # Build smooth curve for the best fit to send back to the frontend
    x_fit = np.linspace(0, 1, 150)
    pdf = candidates[best_name][0]
    y_fit = pdf(x_fit, *best_params)

    return {
        "best_distribution": best_name,
        "p_value": None,  # no KS-test here; using SSE instead
        "parameters": best_params.tolist(),
        "fit_curve": {
            "x": x_fit.tolist(),
            "y": y_fit.tolist(),
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