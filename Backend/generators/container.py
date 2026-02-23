# generator_container.py
# Autor: CHAIMAA KARIOUI

"""
Projekt: SynthData Wizard
Datei: generator_container.py
Autor: CHAIMAA KARIOUI

Beschreibung:
Dieses Modul erzeugt synthetische Containerdaten für den Use Case "Logistik".
Die Funktion generate_containerData baut dynamisch einen DataFrame anhand der vom
Frontend übergebenen Felddefinitionen (rows: List[FrontendField]) auf.

Klassendokumentation (Modul-/Komponentenübersicht):
- Dieses Modul enthält keine Klassen, sondern Funktionen.
- Zentrale Funktion:
  - generate_containerData(rows, rowCount): erzeugt Containerdaten inkl. optionaler Carrier-Integration.
- Hilfsfunktionen:
  - _apply_time_dependencies: erzeugt konsistente timeIn/timeOut Werte basierend auf ETA/ETD.
  - _apply_container_dependencies: setzt Container-Attribute in Abhängigkeit von Carrier-Infos.
  - _calculate_final_dependencies: berechnet abgeleitete Felder (z. B. Gewicht, Status, dwelltime).
  - _categorical_exact: erzeugt kategoriale Werte mit (nahezu) exakter Gewichtsverteilung.
  - _split_list: Hilfsparser für kommaseparierte Strings.

Implementierungsdokumentation (Designentscheidungen):
- Die Spaltennamen sind frei wählbar (Frontend-"name"), die Logik läuft über "type".
- Verteilungen werden aus distributionConfig gesteuert (z. B. uniform/normal/categorical).
- Für Kategorien kann zusätzlich valueSource="custom" mit customValues genutzt werden.
- Wenn Carrier- oder Zeit-Felder angefragt werden, werden Carrier-Daten über generate_carrierData integriert.
- Abhängigkeiten werden in drei Stufen angewendet:
  1) Direktgenerierung aus Verteilungen bzw. Defaults
  2) Carrier-Integration + Zeit-/Container-Abhängigkeiten
  3) Finale Ableitungen (z. B. dwelltime aus timeIn/timeOut)
"""

from typing import List, Optional
import pandas as pd
import numpy as np
import random
from generators.carrier import generate_carrierData
from field_schemas import FrontendField


def generate_containerData(rows: List[FrontendField], rowCount: int) -> pd.DataFrame:
    """
    Zweck:
    Erzeugt einen DataFrame mit synthetischen Containerdaten basierend auf
    den vom Frontend definierten Feldern (rows).

    Args:
        rows: Liste von FrontendField-Objekten (Felddefinitionen aus dem Frontend).
              Wichtige Attribute pro Feld:
              - name: Spaltenname (frei wählbar)
              - type: Feldtyp (steuert Logik, z. B. "unitname", "containertyp", ...)
              - distributionConfig: Verteilungsparameter (distribution, parameterA/B, extraParams)
              - valueSource: "default" | "custom"
              - customValues: Liste eigener Werte (bei valueSource="custom")
        rowCount: Anzahl der zu erzeugenden Datensätze (Zeilen).

    Returns:
        pd.DataFrame mit rowCount Zeilen und den angeforderten Spalten.

    Implementierungsdokumentation (Ablauf):
    1) Feldanforderungen normalisieren (reqs):
       - Spaltenname: name (Fallback auf type)
       - Feldtyp: lowercased (für robuste Vergleiche)
       - distributionConfig / valueSource / customValues übernehmen

    2) Leeren DataFrame mit Index 0..rowCount-1 erstellen.

    3) Für jeden angeforderten Feldtyp Werte erzeugen:
       - unitname: eindeutige Container-ID
       - containertyp, attributestatus, attributedirection: kategorisch oder per customValues/default
       - attributesize: uniform/categorical/default
       - attributeweight: uniform/normal oder später abhängig (NA)
       - timein/timeout: uniform über Datumsbereich oder später abhängig (NA)
       - dwelltime: uniform/normal oder später aus timein/timeout (NA)

    4) Falls Carrier-/Zeitfelder benötigt werden:
       - generate_carrierData erzeugt Carrier-Stammdaten
       - zufällige Zuordnung von Carrier-ID pro Containerzeile
       - Mapping der Carrier-Felder in Container-DataFrame
       - _apply_time_dependencies (timeIn/timeOut konsistent zu ETA/ETD)
       - _apply_container_dependencies (z. B. Typ/Größe abhängig von Route/TEU)

    5) Final:
       - _calculate_final_dependencies (Gewicht, Status, dwelltime etc. falls noch NA)
    """
    rng = np.random.default_rng()

    # gewünschte Spalten (Name beliebig, Logik über type)
    reqs = []
    for r in (rows or []):
        name = (getattr(r, "name", "") or "").strip()
        ftype = (getattr(r, "type", "") or "").strip().lower()
        dist_config = getattr(r, "distributionConfig", None)
        valueSource = getattr(r, "valueSource", None)
        customValues = getattr(r, "customValues", None)
        reqs.append(
            {
                "name": name or ftype,
                "type": ftype,
                "dist_config": dist_config,
                "valueSource": valueSource,
                "customValues": customValues,
            }
        )

    # Leeren DataFrame erstellen
    df = pd.DataFrame(index=range(rowCount))

    # ---------------- Container-Stammdaten mit Verteilungen ----------------
    CONTAINER_TYPE_CHOICES = ["Standard", "High Cube", "Reefer", "Flat Rack", "Open Top"]
    CONTAINER_SIZES = [20.0, 40.0, 45.0]

    def random_container_id() -> str:
        """
        Zweck:
        Erzeugt eine synthetische, menschenlesbare Container-ID.

        Returns:
            String im Format:
            CNT-<5-stellige Zahl>-<3 Zeichen>

        Implementierung:
        - 3 Zeichen werden aus einem "verwechslungsarmen" Alphabet gezogen (ohne I/O/1/0).
        - 5-stellige Nummer wird per rng.integers erzeugt.
        """
        alphabet = list("ABCDEFGHJKLMNPQRSTUVWXYZ23456789")
        random_chars = "".join(rng.choice(alphabet, size=3))
        return f"CNT-{rng.integers(10000,99999)}-{random_chars}"

    # Container-Basisdaten mit Verteilungsunterstützung
    for r in reqs:
        col_name = r["name"]
        field_type = r["type"]
        dist_config = r["dist_config"]
        valueSource = r["valueSource"]
        customValues = r["customValues"]

        dist = (dist_config.distribution or "").strip().lower() if dist_config else ""

        # 1) UnitName
        if field_type == "unitname":
            df[col_name] = [random_container_id() for _ in range(rowCount)]

        # 2) ContainerTyp
        elif field_type == "containertyp":
            if dist == "categorical":
                paramA = dist_config.parameterA if dist_config else None
                paramB = dist_config.parameterB if dist_config else None

                values = _split_list(paramA) or CONTAINER_TYPE_CHOICES
                # Hinweis: Gewichte sind optional; bei Fehlern/Inkonsistenz wird auf 1.0 defaultet.
                weights = [float(w) for w in _split_list(paramB or "")] if paramB else []

                if (not weights) or (len(weights) != len(values)):
                    weights = [1.0] * len(values)

                df[col_name] = _categorical_exact(values, np.array(weights), rowCount)
            else:
                if valueSource == "custom" and customValues:
                    df[col_name] = [random.choice(customValues) for _ in range(rowCount)]
                else:
                    df[col_name] = [rng.choice(CONTAINER_TYPE_CHOICES) for _ in range(rowCount)]

        # 3) Containergröße (attributeSize)
        elif field_type == "attributesize":
            if dist == "uniform":
                paramA = float(dist_config.parameterA) if dist_config else 20.0
                paramB = float(dist_config.parameterB) if dist_config else 45.0
                df[col_name] = rng.uniform(paramA, paramB, size=rowCount).round(1)

            elif dist == "categorical":
                paramA = dist_config.parameterA if dist_config else None
                paramB = dist_config.parameterB if dist_config else None

                values = _split_list(paramA) or [str(x) for x in CONTAINER_SIZES]
                values = [float(v) for v in values]  # Convert to float
                weights = [float(w) for w in _split_list(paramB or "")] if paramB else []

                if (not weights) or (len(weights) != len(values)):
                    weights = [1.0] * len(values)

                df[col_name] = _categorical_exact(values, np.array(weights), rowCount)

            else:
                if valueSource == "custom" and customValues:
                    df[col_name] = [random.choice(customValues) for _ in range(rowCount)]
                else:
                    df[col_name] = rng.choice(CONTAINER_SIZES, size=rowCount)

        # 4) Containergewicht (attributeWeight)
        elif field_type == "attributeweight":
            paramA = float(dist_config.parameterA) if dist_config else None
            paramB = float(dist_config.parameterB) if dist_config else None

            if dist == "uniform":
                low = paramA if paramA is not None else 2000.0
                high = paramB if paramB is not None else 5000.0
                df[col_name] = rng.uniform(low, high, size=rowCount).round(0)

            elif dist == "normal":
                mean = paramA if paramA is not None else 3500.0
                sd = paramB if paramB is not None else 800.0
                weights = rng.normal(mean, sd, size=rowCount)
                weights = np.clip(weights, 1500.0, 6000.0)  # Realistische Grenzen
                df[col_name] = weights.round(0)

            else:
                # Abhängigkeit: Gewicht wird später aus Größe & Typ abgeleitet.
                df[col_name] = pd.NA

        # 5) Containerstatus (attributeStatus)
        elif field_type == "attributestatus":
            if dist == "categorical":
                paramA = dist_config.parameterA if dist_config else None
                paramB = dist_config.parameterB if dist_config else None

                values = _split_list(paramA) or ["loaded", "empty"]
                weights = [float(w) for w in _split_list(paramB or "")] if paramB else []

                if (not weights) or (len(weights) != len(values)):
                    weights = [1.0] * len(values)

                df[col_name] = _categorical_exact(values, np.array(weights), rowCount)
            else:
                if valueSource == "custom" and customValues:
                    df[col_name] = [random.choice(customValues) for _ in range(rowCount)]
                else:
                    # Abhängigkeit: Status wird später ggf. aus Gewicht abgeleitet.
                    df[col_name] = pd.NA

        # 6) Richtung (attributeDirection)
        elif field_type == "attributedirection":
            if dist == "categorical":
                paramA = dist_config.parameterA if dist_config else None
                paramB = dist_config.parameterB if dist_config else None

                values = _split_list(paramA) or ["import", "export"]
                weights = [float(w) for w in _split_list(paramB or "")] if paramB else []

                if (not weights) or (len(weights) != len(values)):
                    weights = [1.0] * len(values)

                df[col_name] = _categorical_exact(values, np.array(weights), rowCount)
            else:
                if valueSource == "custom" and customValues:
                    df[col_name] = [random.choice(customValues) for _ in range(rowCount)]
                else:
                    df[col_name] = [rng.choice(["import", "export", "transshipment"]) for _ in range(rowCount)]

        # 7) timeIn
        elif field_type == "timein":
            if dist == "uniform" and dist_config:
                paramA = dist_config.parameterA or "2025-01-01"
                paramB = dist_config.parameterB or "2025-12-31"
                start = pd.to_datetime(paramA)
                end = pd.to_datetime(paramB)

                time_deltas = rng.uniform(0, (end - start).total_seconds(), size=rowCount)
                times = [start + pd.Timedelta(seconds=td) for td in time_deltas]
                df[col_name] = [t.strftime("%Y-%m-%d %H:%M:%S") for t in times]
            else:
                # Abhängigkeit: wird später aus ETA/ETD erzeugt.
                df[col_name] = pd.NA

        # 8) timeOut
        elif field_type == "timeout":
            if dist == "uniform" and dist_config:
                paramA = dist_config.parameterA or "2025-01-01"
                paramB = dist_config.parameterB or "2025-12-31"
                start = pd.to_datetime(paramA)
                end = pd.to_datetime(paramB)

                time_deltas = rng.uniform(0, (end - start).total_seconds(), size=rowCount)
                times = [start + pd.Timedelta(seconds=td) for td in time_deltas]
                df[col_name] = [t.strftime("%Y-%m-%d %H:%M:%S") for t in times]
            else:
                # Abhängigkeit: wird später aus ETA/ETD erzeugt.
                df[col_name] = pd.NA

        # 9) dwelltime
        elif field_type == "dwelltime":
            paramA = float(dist_config.parameterA) if dist_config else None
            paramB = float(dist_config.parameterB) if dist_config else None

            if dist == "uniform":
                low = paramA if paramA is not None else 6.0
                high = paramB if paramB is not None else 72.0
                df[col_name] = rng.uniform(low, high, size=rowCount).round(2)

            elif dist == "normal":
                mean = paramA if paramA is not None else 24.0
                sd = paramB if paramB is not None else 12.0
                dwell_times = rng.normal(mean, sd, size=rowCount)
                dwell_times = np.clip(dwell_times, 1.0, 168.0)  # 1h-7 Tage
                df[col_name] = dwell_times.round(2)

            else:
                # Abhängigkeit: wird später aus timeIn/timeOut berechnet.
                df[col_name] = pd.NA

    # ---------------- Carrier-Daten Integration ----------------
    carrier_field_types = {
        "shipname",
        "linername",
        "service_route",
        "servicename",
        "eta",
        "etd",
        "loadteu",
        "dischargeteu",
        "length_m",
    }

    needs_carrier = any(r["type"] in carrier_field_types for r in reqs)
    needs_times = any(r["type"] in {"timein", "timeout", "dwelltime"} for r in reqs)

    if needs_carrier or needs_times:
        n_carriers = max(5, min(50, rowCount // 3))
        carr = generate_carrierData(n_carriers)
        carr_map = carr.set_index("id").to_dict("index")
        carrier_ids = rng.choice(carr["id"].tolist(), size=rowCount, replace=True)

        def get_carrier_value(cid, field):
            return carr_map.get(cid, {}).get(field)

        carrier_mapping = {
            "shipname": "shipName",
            "linername": "linerName",
            "service_route": "serviceRoute",
            "servicename": "serviceName",
            "eta": "eta",
            "etd": "etd",
            "loadteu": "loadTEU",
            "dischargeteu": "dischargeTEU",
            "length_m": "length_m",
        }

        for r in reqs:
            if r["type"] in carrier_mapping:
                if r["valueSource"] == "custom" and r["customValues"]:
                    df[r["name"]] = [random.choice(r["customValues"]) for _ in range(rowCount)]
                else:
                    df[r["name"]] = [
                        get_carrier_value(cid, carrier_mapping[r["type"]]) for cid in carrier_ids
                    ]

        _apply_time_dependencies(df, reqs, carrier_ids, carr_map, rng)
        _apply_container_dependencies(df, reqs, carrier_ids, carr_map, rng)

    # ---------------- Finale Abhängigkeiten berechnen ----------------
    _calculate_final_dependencies(df, reqs, rng)

    return df


def _apply_time_dependencies(df, reqs, carrier_ids, carr_map, rng):
    """
    Zweck:
    Erzeugt konsistente Zeitwerte (timeIn/timeOut) anhand der Carrier-Zeiten (ETA/ETD),
    sofern diese Spalten angefragt wurden und noch nicht befüllt sind (NA).

    Logik:
    - timeIn >= ETA (Container kommt nach / beim Schiff an)
    - timeOut <= ETD (Container geht vor Schiff ab)
    - timeOut > timeIn (positive Verweilzeit)

    Args:
        df: Ziel-DataFrame, in den geschrieben wird.
        reqs: Normalisierte Feldanforderungen (Liste von dicts mit name/type/...).
        carrier_ids: Zugewiesene Carrier-ID pro Containerzeile.
        carr_map: Mapping carrier_id -> Carrier-Datensatz (dict).
        rng: numpy Zufallsgenerator.

    Implementierung (Schritte):
    1) Ermittelt timeIn/timeOut Felder, die noch NA enthalten.
    2) Liest ETA/ETD pro Zeile aus carr_map.
    3) Pro Zeile:
       - timeIn = ETA + [0..36] Stunden (begrenzt durch ETD-ETA)
       - timeOut = ETD - [1..24] Stunden (begrenzt durch ETD-timeIn)
       - Fallback: timeOut = timeIn + 1h, wenn die Grenzen nicht passen
    4) Schreibt die Zeiten als Strings "YYYY-mm-dd HH:MM:SS" in die Spalten.
    """
    def get_carrier_value(cid, field):
        return carr_map.get(cid, {}).get(field)

    timein_fields = [
        r for r in reqs if r["type"] == "timein" and pd.isna(df[r["name"]]).any()
    ]
    timeout_fields = [
        r for r in reqs if r["type"] == "timeout" and pd.isna(df[r["name"]]).any()
    ]

    if timein_fields or timeout_fields:
        eta_times = pd.to_datetime([get_carrier_value(cid, "eta") for cid in carrier_ids])
        etd_times = pd.to_datetime([get_carrier_value(cid, "etd") for cid in carrier_ids])

        time_in, time_out = [], []

        for eta, etd in zip(eta_times, etd_times):
            max_hours_after_eta = min(36, int((etd - eta).total_seconds() / 3600) - 1)
            hours_after_eta = 0 if max_hours_after_eta < 0 else rng.integers(0, max_hours_after_eta + 1)
            t_in = eta + pd.Timedelta(hours=hours_after_eta)
            time_in.append(t_in)

            max_hours_before_etd = min(24, int((etd - t_in).total_seconds() / 3600) - 1)
            hours_before_etd = 1 if max_hours_before_etd < 1 else rng.integers(1, max_hours_before_etd + 1)
            t_out = etd - pd.Timedelta(hours=hours_before_etd)

            if t_out <= t_in:
                t_out = t_in + pd.Timedelta(hours=1)

            time_out.append(t_out)

        for timein_field in timein_fields:
            df[timein_field["name"]] = [t.strftime("%Y-%m-%d %H:%M:%S") for t in time_in]

        for timeout_field in timeout_fields:
            df[timeout_field["name"]] = [t.strftime("%Y-%m-%d %H:%M:%S") for t in time_out]


def _apply_container_dependencies(df, reqs, carrier_ids, carr_map, rng):
    """
    Zweck:
    Setzt Containerattribute in Abhängigkeit von Carrier-Eigenschaften,
    sofern diese Spalten angefragt wurden und noch nicht befüllt sind (NA).

    Abhängigkeiten:
    - attributesize: abhängig von loadTEU (Schiffskapazität)
    - containertyp: abhängig von serviceRoute (Stichwort-basierte Heuristik)

    Args:
        df: Ziel-DataFrame.
        reqs: Feldanforderungen.
        carrier_ids: Zugewiesene Carrier-IDs pro Zeile.
        carr_map: carrier_id -> Carrier-Datensatz.
        rng: numpy Zufallsgenerator.

    Implementierung:
    - Größe:
      - Bei kleinen TEU eher 20ft, bei großen TEU eher 40/45ft (gewichtete Auswahl).
    - Typ:
      - Bei "trop/asia/africa" höhere Reefer-Wahrscheinlichkeit (Kühlcontainer).
      - Bei "heavy/construction/project" eher Flat Rack / Open Top.
      - Sonst Standard/High Cube dominant.
    """
    def get_carrier_value(cid, field):
        return carr_map.get(cid, {}).get(field)

    size_fields = [
        r for r in reqs if r["type"] == "attributesize" and pd.isna(df[r["name"]]).any()
    ]
    if size_fields:
        for field in size_fields:
            sizes = []
            for cid in carrier_ids:
                teu = get_carrier_value(cid, "loadTEU") or 1000
                if teu < 1000:
                    sizes.append(rng.choice([20.0, 40.0], p=[0.7, 0.3]))
                elif teu < 2000:
                    sizes.append(rng.choice([20.0, 40.0, 45.0], p=[0.5, 0.4, 0.1]))
                else:
                    sizes.append(rng.choice([20.0, 40.0, 45.0], p=[0.3, 0.5, 0.2]))
            df[field["name"]] = sizes

    type_fields = [
        r for r in reqs if r["type"] == "containertyp" and pd.isna(df[r["name"]]).any()
    ]
    if type_fields:
        for field in type_fields:
            types = []
            for cid in carrier_ids:
                route = (get_carrier_value(cid, "serviceRoute") or "").lower()
                if any(x in route for x in ["asia", "trop", "africa"]):
                    types.append(rng.choice(["Reefer", "Standard", "High Cube"], p=[0.4, 0.4, 0.2]))
                elif any(x in route for x in ["project", "heavy", "construction"]):
                    types.append(rng.choice(["Flat Rack", "Open Top", "Standard"], p=[0.3, 0.3, 0.4]))
                else:
                    types.append(rng.choice(["Standard", "High Cube", "Reefer"], p=[0.6, 0.3, 0.1]))
            df[field["name"]] = types


def _calculate_final_dependencies(df, reqs, rng):
    """
    Zweck:
    Berechnet abgeleitete/abhängige Containerfelder, falls diese noch nicht
    durch direkte Verteilungen oder Carrier-Abhängigkeiten befüllt wurden.

    Abhängigkeiten:
    1) attributeweight (falls NA):
       - abhängig von attributesize (20/40/45) und containertyp (Reefer/heavy/open)
    2) attributestatus (falls NA):
       - abhängig von attributeweight (empty wenn < 2500, sonst loaded)
    3) dwelltime (falls NA):
       - aus timeIn/timeOut, sofern beide vorhanden und konsistent

    Args:
        df: DataFrame mit ggf. teilweise gefüllten Spalten.
        reqs: Feldanforderungen.
        rng: numpy Zufallsgenerator.

    Implementierung:
    - Sucht passende Spaltennamen im reqs (name der Spalte, deren type passt).
    - Nur Felder, die noch NA enthalten, werden nachträglich berechnet.
    - Fallbacks nutzen realistische Standardbereiche, wenn benötigte Spalten fehlen.
    """
    weight_fields = [
        r for r in reqs if r["type"] == "attributeweight" and pd.isna(df[r["name"]]).any()
    ]
    if weight_fields:
        for field in weight_fields:
            weights = []
            size_col = next((r["name"] for r in reqs if r["type"] == "attributesize"), None)
            type_col = next((r["name"] for r in reqs if r["type"] == "containertyp"), None)

            for i in range(len(df)):
                size = df[size_col].iloc[i] if size_col else 40.0
                container_type = df[type_col].iloc[i] if type_col else "Standard"

                if size == 20.0:
                    base = rng.integers(2000, 2400)
                elif size == 40.0:
                    base = rng.integers(3800, 4300)
                else:
                    base = rng.integers(4800, 5200)

                if "reefer" in str(container_type).lower():
                    base += rng.integers(500, 1000)
                elif "flat" in str(container_type).lower() or "open" in str(container_type).lower():
                    base -= rng.integers(200, 500)

                weights.append(base)

            df[field["name"]] = weights

    status_fields = [
        r for r in reqs if r["type"] == "attributestatus" and pd.isna(df[r["name"]]).any()
    ]
    if status_fields:
        for field in status_fields:
            weight_col = next((r["name"] for r in reqs if r["type"] == "attributeweight"), None)
            if weight_col:
                df[field["name"]] = ["empty" if w < 2500 else "loaded" for w in df[weight_col]]
            else:
                df[field["name"]] = [rng.choice(["loaded", "empty"], p=[0.7, 0.3]) for _ in range(len(df))]

    dwell_fields = [
        r for r in reqs if r["type"] == "dwelltime" and pd.isna(df[r["name"]]).any()
    ]
    if dwell_fields:
        for field in dwell_fields:
            timein_col = next((r["name"] for r in reqs if r["type"] == "timein"), None)
            timeout_col = next((r["name"] for r in reqs if r["type"] == "timeout"), None)

            if timein_col and timeout_col:
                dwell_times = []
                for i in range(len(df)):
                    try:
                        t_in = pd.to_datetime(df[timein_col].iloc[i])
                        t_out = pd.to_datetime(df[timeout_col].iloc[i])
                        if pd.notna(t_in) and pd.notna(t_out) and t_out > t_in:
                            hours = (t_out - t_in).total_seconds() / 3600.0
                            dwell_times.append(round(hours, 2))
                        else:
                            dwell_times.append(round(float(rng.uniform(6.0, 72.0)), 2))
                    except Exception:
                        dwell_times.append(round(float(rng.uniform(6.0, 72.0)), 2))
                df[field["name"]] = dwell_times
            else:
                df[field["name"]] = [round(float(rng.uniform(6.0, 72.0)), 2) for _ in range(len(df))]


# globaler RNG für deterministisches Mischen in _categorical_exact
rng = np.random.default_rng()


def _categorical_exact(values: list, weights: np.ndarray, size: int) -> list:
    """
    Zweck:
    Erzeugt eine Liste kategorialer Werte mit möglichst exakter Verteilung
    gemäß den angegebenen Gewichten.

    Args:
        values: Liste der Kategorien (z. B. ["Standard", "Reefer"]).
        weights: Gewichte pro Kategorie (numpy array, gleiche Länge wie values).
        size: Länge der Ergebnisliste.

    Returns:
        Liste der Länge size, deren Kategorien nach Gewichten verteilt sind.

    Implementierung:
    - Normalisierung der Gewichte (Summe = 1).
    - "Methode des größten Restes":
      1) target = weights * size
      2) counts = floor(target)
      3) Restplätze werden nach absteigendem Nachkommaanteil vergeben
    - Danach wird die Ergebnisliste mit rng.shuffle gemischt, damit die Reihenfolge
      nicht blockweise ist.
    """
    if len(values) == 0:
        return [None] * size
    if weights is None or len(weights) != len(values):
        return [random.choice(values) for _ in range(size)]

    w = np.array(weights, dtype=float)
    if w.sum() <= 0:
        w = np.ones_like(w)
    w = w / w.sum()

    target = w * size
    counts = np.floor(target).astype(int)
    remainder = size - counts.sum()

    if remainder > 0:
        frac = target - np.floor(target)
        idxs = np.argsort(-frac)
        for i in range(remainder):
            counts[idxs[i % len(idxs)]] += 1

    out = []
    for val, c in zip(values, counts):
        out.extend([val] * int(c))

    if len(out) < size:
        out.extend([values[0]] * (size - len(out)))
    elif len(out) > size:
        out = out[:size]

    arr = np.array(out, dtype=object)
    rng.shuffle(arr)
    return arr.tolist()


def _split_list(s: Optional[str]) -> list[str]:
    """
    Zweck:
    Parst einen kommaseparierten String in eine Liste bereinigter Einträge.

    Args:
        s: Optionaler String, z. B. "a, b, c"

    Returns:
        Liste von Strings ohne Leerzeichen und ohne leere Einträge.

    Beispiele:
        _split_list(" 20,40 , 45 ") -> ["20", "40", "45"]
        _split_list(None) -> []

    Implementierung:
    - Split bei Komma
    - strip() pro Element
    - Filter auf nicht-leere Strings
    """
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]