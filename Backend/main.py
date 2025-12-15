# backend/main.py
from fastapi import FastAPI, UploadFile, File, Form
from typing import List
from pydantic import BaseModel
from field_schemas import FrontendField, ExportRequest
import field_storage
from fastapi.middleware.cors import CORSMiddleware
import io
import pandas as pd
from fastapi.responses import StreamingResponse, JSONResponse
from generators.health import generate_healthData
from generators.finance import generate_financeData
from generators.container import generate_containerData
from generators.general import generate_generalData
from scipy import stats
import numpy as np
import re

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
async def receive_fields(fields: List[FrontendField]):
    field_storage.save_fields(fields)
    return {"status": "ok", "received_fields": len(fields)}

@app.get("/api/v1/fields")
async def list_fields():
    return field_storage.get_fields()


# -------------------------
# XLSX Sheet helpers
# -------------------------
def sanitize_sheet_name(name: str) -> str:
    r"""
    Excel-Regeln:
    - max 31 Zeichen
    - keine Zeichen: : \ / ? * [ ]
    - darf nicht leer sein
    """
    if not name:
        return "Sheet"
    cleaned = re.sub(r"[:\\/?*\[\]]", "", name).strip()
    if not cleaned:
        cleaned = "Sheet"
    return cleaned[:31]


def make_unique_sheet_names(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for n in names:
        base = n
        if base not in seen:
            seen[base] = 1
            out.append(base)
        else:
            seen[base] += 1
            suffix = f" ({seen[base]})"
            trimmed = base[: max(0, 31 - len(suffix))] + suffix
            out.append(trimmed)
    return out


# === Export-Endpunkt ===
@app.post("/api/export")
async def export_data(request: ExportRequest):
    print(request)
    usedUseCaseIds = request.usedUseCaseIds or []

    df_list = []
    for ucid in usedUseCaseIds:
        if ucid.lower() == "logistik":
            temp_df = generate_containerData(request.rows, request.rowCount)
            df_list.append(temp_df)
        elif ucid.lower() == "gesundheit":
            temp_df = generate_healthData(request.rows, request.rowCount)
            df_list.append(temp_df)
        elif ucid.lower() == "finanzen":
            temp_df = generate_financeData(request.rows, request.rowCount)
            df_list.append(temp_df)
        elif ucid.lower() == "general":
            temp_df = generate_generalData(request.rows, request.rowCount)
            df_list.append(temp_df)
    
    if df_list:
        df = pd.concat(df_list, ignore_index=True)
    else:
        df = pd.DataFrame()
            
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
            sql_buffer.write(
                f"INSERT INTO {table_name} ({columns}) VALUES ({', '.join(values)});\n"
            )
        sql_buffer.seek(0)
        return StreamingResponse(
            sql_buffer,
            media_type="application/sql",
            headers={"Content-Disposition": "attachment; filename=synthdata.sql"},
        )

    # === XLSX Export ===
    elif fmt == "XLSX":
        output = io.BytesIO()

        # Fallback: wenn keine Sheets kommen -> 1 Sheet
        if not request.sheets:
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Daten")
            output.seek(0)
            return StreamingResponse(
                output,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": "attachment; filename=synthdatawizard.xlsx"},
            )

        # 1) Namen säubern + unique machen
        raw_names = [sanitize_sheet_name(s.name) for s in request.sheets]
        unique_names = make_unique_sheet_names(raw_names)

        # 2) Pro Sheet: nur die ausgewählten Spalten exportieren
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            for sheet_cfg, sheet_name in zip(request.sheets, unique_names):
                wanted = [c for c in sheet_cfg.fieldNames if c in df.columns]

                # Debug-Hilfe (falls leer)
                if len(wanted) == 0:
                    print("WARN: Sheet hat keine passenden Spalten:", sheet_cfg.name)
                    print("wanted fieldNames:", sheet_cfg.fieldNames)
                    print("df.columns:", list(df.columns))

                    # Optional: leeres Sheet schreiben
                    pd.DataFrame().to_excel(writer, index=False, sheet_name=sheet_name)
                    continue

                df_sheet = df[wanted].copy()
                df_sheet.to_excel(writer, index=False, sheet_name=sheet_name)

        output.seek(0)
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=synthdatawizard.xlsx"},
        )

    # === CSV Export (Standard) ===
    else:
        separator = "," if fmt == "CSV" else ";"
        line_end = "\r\n" if "CRLF" in (request.lineEnding or "").upper() else "\n"

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


# ==== Custom Distribution Fitting API ====
class DistributionFitRequest(BaseModel):
    points: list[float]


@app.post("/api/fit-distribution")
async def fit_distribution(request: DistributionFitRequest):
    points = np.array(request.points, dtype=float)
    points = points[np.isfinite(points)]
    if len(points) < 2:
        return JSONResponse(status_code=400, content={"error": "Not enough points"})

    min_val = np.min(points)
    max_val = np.max(points)
    if max_val - min_val > 0:
        norm_points = (points - min_val) / (max_val - min_val)
    else:
        norm_points = np.zeros_like(points)

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
    return load_profiles()

@app.post("/profiles")
def create_profile(profile: dict):
    name = profile.get("name")
    if not name:
        return {"error": "Name ist erforderlich"}
    new_profile = add_profile(name)
    return new_profile

@app.delete("/profiles/{profile_id}")
def remove_profile(profile_id: str):
    delete_profile(profile_id)
    return {"message": f"Profil {profile_id} wurde gelöscht"}

@app.post("/profiles/{profile_id}/data")
def save_profile_data_route(profile_id: str, data: dict):
    try:
        save_profile_data(profile_id, data)
        return {"message": f"Daten für Profil {profile_id} wurden gespeichert"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/profiles/{profile_id}/data")
def get_profile_data_route(profile_id: str):
    try:
        data = get_profile_data(profile_id)
        if data is None:
            return {"data": {}}
        return {"data": data}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
