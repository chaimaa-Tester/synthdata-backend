from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from typing import List, Literal
from field_schemas import FieldDefinition, ExportRequest
import field_storage
from fastapi.middleware.cors import CORSMiddleware
import io
import pandas as pd
from fastapi.responses import StreamingResponse, JSONResponse
from generator_health import generate_healthData
from generator_finance import generate_financeData
from generator_container import generate_containerData
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

# # SQL Dialekt-Typen als Konstanten
# SUPPORTED_SQL_DIALECTS = Literal["mysql", "postgresql"]

@app.post("/api/export")
async def export_csv(request: ExportRequest):
    # Debug: Request-Objekt ausgeben (kann entfernt/auf logging umgestellt werden)
    print(request)

    # Liste der ausgewählten UseCases (sicher machen, dass None abgefangen wird)
    usedUseCaseIds = request.usedUseCaseIds or []

    # Wir sammeln alle Teilergebnisse (DataFrames) der einzelnen Generatoren in einer Liste.
    # Warum? Wenn mehrere UseCases gewählt werden, erzeugt jeder Generator eigene Spalten.
    # Überschreiben würde vorherige Ergebnisse verlieren; am Ende führen wir zusammen.
    dfs: list[pd.DataFrame] = []

    # Mapping von UseCase-ID (lowercase) auf die Generator-Funktion
    usecase_map = {
    "logistik": generate_containerData,   # Container + Carrier integriert
    "gesundheit": generate_healthData,
    "finanzen": generate_financeData,
}


    for ucid in usedUseCaseIds:
        if not ucid:
            continue
        gen = usecase_map.get(ucid.lower())
        if gen:
            # Generator aufrufen; erwartet (rows, rowCount) und liefert ein DataFrame
            df_part = gen(request.rows, request.rowCount)
            if isinstance(df_part, pd.DataFrame) and not df_part.empty:
                dfs.append(df_part)
        else:
            # Unbekannter UseCase -> Warnung (kein Abbruch)
            print(f"WARNUNG: Unbekannter UseCase '{ucid}'")

    # Wenn kein Generator gültige Daten geliefert hat, Fehler zurückgeben
    if not dfs:
        raise HTTPException(status_code=400, detail="Keine gültigen UseCases/Generatoren ausgewählt oder keine Daten erzeugt")

    # Falls mehrere DataFrames: zusammenfügen (outer-join mittels concat, unterschiedliche Spalten bleiben erhalten)
    if len(dfs) == 1:
        df = dfs[0]
    else:
        # Wenn mehrere Generatoren gewählt wurden, die Ergebnisse spaltenweise kombinieren
        # (jede Generator liefert die gleichen Anzahl Zeilen für die Vorschau/export)
        df = pd.concat(dfs, axis=1)
        # Duplikate in Spaltennamen vermeiden: erste Vorkommen behalten
        df = df.loc[:, ~df.columns.duplicated()]

    # nach dem Zusammenfügen aller Generator-DataFrames -> df vorhanden
    # Wunsch: nur die Felder exportieren, die im Frontend ausgewählt wurden (request.rows)
    requested = [ (r.name or "").strip() for r in (request.rows or []) if (r.name or "").strip() ]
    if requested:
        # case-insensitive Matching: finde für jeden requested-name die passende Spalte im df
        cols_found = []
        cols_lower = {c.lower(): c for c in df.columns}
        for req in requested:
            rq = req.lower()
            # exaktes Match
            if rq in cols_lower:
                cols_found.append(cols_lower[rq])
                continue
            # substring match (z.B. "unit" findet "unitName")
            matched = None
            for c in df.columns:
                if rq == c.lower() or rq in c.lower() or c.lower() in rq:
                    matched = c
                    break
            if matched:
                cols_found.append(matched)
        # nur gefundene Spalten behalten (fallback: gesamte df, wenn none matched)
        if cols_found:
            df = df[cols_found]

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
        # JSON (pretty, records) — use BytesIO and proper media type to ensure a .json download
        json_str = df.to_json(orient="records", force_ascii=False, indent=2)
        json_bytes = json_str.encode("utf-8")
        json_buffer = io.BytesIO(json_bytes)        
        json_buffer.seek(0)
        return StreamingResponse(
            json_buffer,
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=synthdata.json"},
        )
    
    # elif request.format.upper() == "SQL":
    #     # Validierung der SQL-Konfiguration
    #     if not request.sqlConfig:
    #         raise HTTPException(status_code=400, detail="SQL-Konfiguration fehlt")

    #     table_name = request.sqlConfig.get("tableName", "synthetic_data")
    #     sql_dialect = request.sqlConfig.get("dialect", "mysql").lower()

    #     # Überprüfung des SQL-Dialekts
    #     if sql_dialect not in ["mysql", "postgresql"]:
    #         raise HTTPException(
    #             status_code=400, 
    #             detail=f"Nicht unterstützter SQL-Dialekt: {sql_dialect}"
    #         )

    #     output = io.StringIO()

    #     # Tabellen-Create-Statement
    #     try:
    #         if sql_dialect == "mysql":
    #             create_table = generate_mysql_table(df, table_name)
    #         else:
    #             create_table = generate_postgresql_table(df, table_name)

    #         output.write(f"{create_table}\n\n")

    #         # Werte korrekt für SQL formatieren
    #         def escape_sql_value(v):
    #             if pd.isna(v):
    #                 return "NULL"
    #             if isinstance(v, (int, float)):
    #                 return str(v)
    #             s = str(v).replace("'", "''")  # einfache Quotes escapen
    #             return f"'{s}'"

    #         # Spaltennamen je nach SQL-Dialekt
    #         if sql_dialect == "mysql":
    #             column_names = ", ".join([f"`{col}`" for col in df.columns])
    #         else:
    #             column_names = ", ".join([f'"{col}"' for col in df.columns])

    #         # INSERT-Befehle
    #         for _, row in df.iterrows():
    #             values = [escape_sql_value(v) for v in row]
    #             value_string = ", ".join(values)
    #             output.write(f"INSERT INTO {table_name} ({column_names}) VALUES ({value_string});\n")

    #         output.seek(0)

    #         return StreamingResponse(
    #             output,
    #             media_type="application/sql",
    #             headers={"Content-Disposition": f"attachment; filename={table_name}.sql"},
    #         )
    #     except Exception as e:
    #         raise HTTPException(
    #             status_code=500,
    #             detail=f"Fehler beim Generieren des SQL-Exports: {str(e)}"
    #         )
        
    elif request.format.upper() == "CSV":
        # CSV konfigurieren
        separator = ","
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