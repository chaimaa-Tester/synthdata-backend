"""
--------------------------------------------------------------------
Projekt: SynthData Wizard
Datei: main.py
Autoren: Burak Arabaci

Beschreibung:
Dieser Code-Ausschnitt implementiert zentrale Backend-Funktionen für die
SynthData-Wizard-Anwendung. Er umfasst:

1) Profile Management:
   - Laden, Erstellen und Löschen von Profilen
   - Speichern und Abrufen profilbezogener Konfigurationsdaten

2) Export-Funktion:
   - Generierung synthetischer Datensätze basierend auf Use-Cases und Felddefinitionen
   - Spezielle Behandlung für Namensfelder über eine interne Name-Source-API
   - Export der generierten Daten in JSON / SQL / XLSX / CSV

3) Erkennung der Custom Verteilungen:
   - Generierung von Verteilungen anhand einer Zeichnung
   - Erkennen einer Verteilung aus einem bestimmten Auszug einer hochgeladenen Datei

Technische Details:
- Framework: FastAPI
- Datenverarbeitung: pandas
- Streaming Downloads: StreamingResponse
- Fallback-Strategie: robuste Namensgenerierung bei API-Fehlern

Architekturprinzip:
- Trennung der Verantwortlichkeiten:
  * Profile-spezifische Persistenz erfolgt über storage_manager
  * Fachliche Datengeneratoren liegen in generators/*
  * Namensbeschaffung über interne API (name_source_router) + Fallback
--------------------------------------------------------------------
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from scipy import stats
import io
import pandas as pd
import re

# Generatoren für die jeweiligen Use-Cases
from generators.health import generate_healthData
from generators.finance import generate_financeData
from generators.container import generate_containerData
from generators.general import generate_generalData

# Request-/Field-Schemas (Frontend -> Backend Contract)
from field_schemas import FrontendField, ExportRequest
import field_storage
from typing import List

# Profil-Persistenz (JSON-basiert) wird ausgelagert verwaltet
from storage_manager import (
    load_profiles,
    add_profile,
    delete_profile,
    save_profile_data,
    get_profile_data,
)

# Für Name-Source API Calls + Fallback
import requests

# Für die Erstellung eines Verbindungsobjekts zur DB
from dbhandler import connect_to_DB


app = FastAPI()
# === CORS-Konfiguration ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Routing ===
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

# ============================================================
# Hilfsfunktionen: Namensgenerierung (Name-Source + Fallback)
# ============================================================

async def generate_name_values(field_type: str, name_source: str, country: str, count: int):
    """
    Generiert Namenswerte über die interne Name-Source API.

    Zweck:
    - Unterstützt konfigurierbare Namensquellen (z.B. "western" oder "regional")
    - Liefert Listen mit Namen für Felder vom Typ: vorname, nachname, name

    Robustheit:
    - Bei API-Fehlern wird ein Fallback via mimesis verwendet, damit der Export
      nicht abbricht und weiterhin nutzbar bleibt.

    Parameter:
    - field_type: "vorname" | "nachname" | "name"
    - name_source: "western" oder "regional" (aus distributionConfig)
    - country: Land (nur relevant bei regionaler Quelle)
    - count: Anzahl der zu generierenden Werte

    Rückgabe:
    - Liste[str] mit generierten Namen
    """
    try:
        # Mapping: je nach Feldtyp werden unterschiedliche Namenswerte erzeugt
        if field_type == "vorname":
            name_field_type = "vorname"
        elif field_type == "nachname":
            name_field_type = "nachname"
        else:
            name_field_type = "name"

        # Interner API-Call (Name-Source Endpoint)
        response = requests.post(
            "http://localhost:8000/api/name-source/generate",
            json={
                "source_type": "western" if name_source == "western" else "regional",
                "country": country if name_source == "regional" else None,
                "name_field_type": name_field_type,
                "count": count,
                "gender": None,  # optional erweiterbar (z.B. aus Felddaten)
            },
        )

        if response.status_code == 200:
            data = response.json()
            return data.get("names", [])

        print(f"Name-Source API Fehler: {response.status_code}")
        return generate_fallback_names(field_type, count)

    except Exception as e:
        print(f"Fehler bei Namensgenerierung: {e}")
        return generate_fallback_names(field_type, count)


def generate_fallback_names(field_type: str, count: int):
    """
    Fallback-Mechanismus: Generiert einfache Namen mit mimesis.

    Motivation:
    - Garantiert Funktionsfähigkeit des Exports auch bei API-Ausfällen
    - Verhindert "hard failures" im Gesamtablauf
    """
    from mimesis import Person

    person = Person()

    if field_type == "vorname":
        return [person.first_name() for _ in range(count)]
    elif field_type == "nachname":
        return [person.last_name() for _ in range(count)]
    else:
        return [person.full_name() for _ in range(count)]


# ============================================================
# Export-Endpunkt: Generierung + Ausgabe (JSON/SQL/XLSX/CSV)
# ============================================================

@app.post("/export")
async def export_data(request: ExportRequest):
    """
    Exportiert synthetische Daten anhand eines ExportRequest.

    Ablauf (High-Level):
    1) Identifikation von Namensfeldern (vorname/nachname/name) inkl. Config
    2) Vorab-Generierung von Namenswerten über Name-Source API (effizient gebündelt)
    3) Generierung der restlichen Daten je Use-Case mittels generators/*
    4) Überschreiben/Einsetzen der Namensspalten im finalen DataFrame
    5) Export im gewünschten Format:
       - JSON, SQL, XLSX oder CSV

    Hinweis:
    - request.usedUseCaseIds steuert, welche Use-Case-Generatoren genutzt werden
    - request.rows definieren die Spalten/Felder
    - request.rowCount definiert die Anzahl Datensätze
    """
    print("Export-Request empfangen:", request)

    usedUseCaseIds = request.usedUseCaseIds or []

    # 1) Namensfelder identifizieren (inkl. DistributionConfig)
    name_fields_to_generate = []
    for row in request.rows:
        if row.type in ["vorname", "nachname", "name"]:
            dist_config = getattr(row, "distributionConfig", {}) or {}
            name_source = dist_config.get("name_source")
            country = dist_config.get("country")

            # Nur wenn tatsächlich konfiguriert, wird Name-Source genutzt
            if name_source:
                name_fields_to_generate.append({
                    "row_index": request.rows.index(row),
                    "field_type": row.type,
                    "name_source": name_source,
                    "country": country,
                    "field_name": row.name,
                })

    # 2) Namenswerte generieren
    generated_values_map = {}
    for name_field in name_fields_to_generate:
        values = await generate_name_values(
            field_type=name_field["field_type"],
            name_source=name_field["name_source"],
            country=name_field["country"],
            count=request.rowCount,
        )
        generated_values_map[name_field["row_index"]] = values

    # 3) Synthetische Daten je Use-Case generieren und zusammenführen
    df_list = []
    for ucid in usedUseCaseIds:
        uc = (ucid or "").lower()

        if uc == "logistik":
            df_list.append(generate_containerData(request.rows, request.rowCount))
        elif uc == "gesundheit":
            df_list.append(generate_healthData(request.rows, request.rowCount))
        elif uc == "finanzen":
            df_list.append(generate_financeData(request.rows, request.rowCount))
        elif uc == "general":
            df_list.append(generate_generalData(request.rows, request.rowCount))

    df = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

    # 4) Namenswerte in DataFrame einsetzen (wenn Spalte existiert)
    for row_idx, values in generated_values_map.items():
        if row_idx < len(request.rows):
            field_name = request.rows[row_idx].name

            if field_name in df.columns and len(values) == len(df):
                df[field_name] = values
            elif field_name in df.columns:
                # Falls Werteliste zu kurz: wiederholen (robust gegen Edge-Cases)
                if len(values) < len(df) and len(values) > 0:
                    repeated = (values * (len(df) // len(values) + 1))[:len(df)]
                    df[field_name] = repeated
                else:
                    df[field_name] = values[:len(df)]

    fmt = request.format.upper()

    # ----------------------------
    # JSON Export
    # ----------------------------
    if fmt == "JSON":
        json_str = df.to_json(orient="records", force_ascii=False, indent=2)
        buf = io.BytesIO(json_str.encode("utf-8"))
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=synthdata.json"},
        )

    # ----------------------------
    # SQL Export
    # ----------------------------
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
                    # Basic SQL-Escaping für einfache Stringwerte
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

    # ----------------------------
    # XLSX Export
    # ----------------------------
    elif fmt == "XLSX":
        output = io.BytesIO()

        # Wenn keine Sheets konfiguriert sind -> alles in einem Sheet exportieren
        if not request.sheets:
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Daten")

            output.seek(0)
            return StreamingResponse(
                output,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": "attachment; filename=synthdatawizard.xlsx"},
            )

        # Wenn Sheets vorhanden sind, wird pro Sheet eine Spaltenauswahl exportiert
        # (sanitize_sheet_name / make_unique_sheet_names existieren im Hauptfile)
        raw_names = [sanitize_sheet_name(s.name) for s in request.sheets]
        unique_names = make_unique_sheet_names(raw_names)

        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            for sheet_cfg, sheet_name in zip(request.sheets, unique_names):
                wanted = [c for c in sheet_cfg.fieldNames if c in df.columns]

                if len(wanted) == 0:
                    # Debug- und Robustheitsmaßnahme: leeres Sheet anlegen
                    print("WARN: Sheet hat keine passenden Spalten:", sheet_cfg.name)
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

    # ----------------------------
    # CSV Export (Default)
    # ----------------------------
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

# ============================================================
# Profile Management (DB Verbindungsmanagement über storage_manager)
# Überarbeitet: Jan Krämer
# ============================================================

@app.get("/profiles")
def get_profiles():
    """
    Lädt alle gespeicherten Profile aus der DB.
    """
    if not connect_to_DB():
        return JSONResponse(status_code=503, content={"error": "DB nicht verbunden!"})
    return load_profiles(connect_to_DB())

@app.post("/profiles")
def create_profile(profile: dict):
    """
    Erstellt ein neues Profil und speichert es in der DB.
    
    :param profile: Beschreibung
    :type profile: dict
    """
    name = profile.get("name")
    if not name:
        return {"error": "Name ist erforderlich!"}
    if not connect_to_DB():
        return JSONResponse(status_code=503, content={"error": "DB nicht verbunden!"})
    new_profile = add_profile(connect_to_DB(), name)
    return new_profile
        

@app.delete("/profiles/{db_id}")
def remove_profile(db_id: str):
    """
    Löscht bestehendes Profil aus der DB.
    
    :param db_id: Die ID des zu löschenden Profils aus der DB.
    :type db_id: str
    """
    if not connect_to_DB():
        return JSONResponse(status_code=503, content={"error": "DB nicht verbunden!"})
    delete_profile(connect_to_DB(), db_id)
    return {"message": f"Profil {db_id} wurde gelöscht."}

@app.post("/profiles/{db_id}/data")
def save_profile_data_route(db_id: str, data: dict):
    """
    Speichert die eingegebenen Daten innerhalb eines Profils.
    
    :param db_id: Die zugehörige ID des Profils.
    :type db_id: str
    :param data: Die eingegeben Daten aus der UI.
    :type data: dict
    """
    if not connect_to_DB():
        return JSONResponse(status_code=503, content={"error": "DB nicht verbunden!"})
    try:
        save_profile_data(connect_to_DB(), db_id, data)
        return {"message": f"Daten für Profil {db_id} gespeichert."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/profiles/{db_id}/data")
def get_profile_data_route(db_id: str):
    """
    Lädt die gespeicherten Daten, für das angegebene Profil, aus der DB.
    
    :param db_id: Beschreibung
    :type db_id: str
    """
    if not connect_to_DB():
        return JSONResponse(status_code=503, content={"error": "DB nicht verbunden!"})
    try:
        data = get_profile_data(connect_to_DB(), db_id)
        if data is None:
            return {"data":{}}
        return {"data": data}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
# ======================
# Distribution Detection
# ======================
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

# ===============================
# Custom Distribution Fitting API
# ===============================
class DistributionFitRequest(BaseModel):
    points: list[float]


@app.post("/fit-distribution")
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