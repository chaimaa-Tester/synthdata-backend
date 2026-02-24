"""
--------------------------------------------------------------------
Projekt: SynthData Wizard
Datei: backend/main.py (Auszug)
Autor: Burak Arabaci
Datum: [Abgabedatum eintragen]

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

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List
import io
import pandas as pd

# Generatoren für die jeweiligen Use-Cases
from generators.health import generate_healthData
from generators.finance import generate_financeData
from generators.container import generate_containerData
from generators.general import generate_generalData

# Request-/Field-Schemas (Frontend -> Backend Contract)
from field_schemas import FrontendField, ExportRequest

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


app = FastAPI()

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

@app.post("/api/export")
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

    # 2) Namenswerte generieren (gebündelt)
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


# ============================================================
# Profile Management (JSON-basierte Persistenz über storage_manager)
# ============================================================

@app.get("/profiles")
def get_profiles():
    """
    Gibt eine Liste aller vorhandenen Profile zurück.
    Datenquelle: storage_manager.load_profiles()
    """
    return load_profiles()


@app.post("/profiles")
def create_profile(profile: dict):
    """
    Erstellt ein neues Profil.

    Erwarteter Body:
    - { "name": "<profilname>" }

    Rückgabe:
    - Profilobjekt (inkl. generierter ID)
    """
    name = profile.get("name")
    if not name:
        return {"error": "Name ist erforderlich"}

    new_profile = add_profile(name)
    return new_profile


@app.delete("/profiles/{profile_id}")
def remove_profile(profile_id: str):
    """
    Löscht ein Profil anhand der profile_id.
    """
    delete_profile(profile_id)
    return {"message": f"Profil {profile_id} wurde gelöscht"}


@app.post("/profiles/{profile_id}/data")
def save_profile_data_route(profile_id: str, data: dict):
    """
    Speichert profilbezogene Konfigurations-/Formulardaten.

    Zweck:
    - Erlaubt pro Profil persistente Einstellungen (z.B. Feldkonfigurationen)
    - Datenstruktur wird bewusst flexibel als dict gehalten
    """
    try:
        save_profile_data(profile_id, data)
        return {"message": f"Daten für Profil {profile_id} wurden gespeichert"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/profiles/{profile_id}/data")
def get_profile_data_route(profile_id: str):
    """
    Liefert gespeicherte Profil-Daten zurück.
    Falls keine Daten existieren, wird ein leeres Objekt zurückgegeben.
    """
    try:
        data = get_profile_data(profile_id)
        if data is None:
            return {"data": {}}
        return {"data": data}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})