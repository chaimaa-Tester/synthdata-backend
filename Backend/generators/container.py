# generator_container.py
from typing import List, Optional
import pandas as pd
import numpy as np
import random
from generators.carrier import generate_carrierData
from field_schemas import FrontendField

def generate_containerData(rows: List[FrontendField], rowCount: int) -> pd.DataFrame:
    rng = np.random.default_rng()

    # gewünschte Spalten (Name beliebig, Logik über type)
    reqs = []
    for r in (rows or []):
        name = (getattr(r, "name", "") or "").strip()
        ftype = (getattr(r, "type", "") or "").strip().lower()
        dist_config = getattr(r, 'distributionConfig', None)
        valueSource = getattr(r, "valueSource", None)
        customValues = getattr(r, "customValues", None)
        reqs.append({
            "name": name or ftype, 
            "type": ftype,
            "dist_config": dist_config
        })

    # Leeren DataFrame erstellen
    df = pd.DataFrame(index=range(rowCount))

    # ---------------- Container-Stammdaten mit Verteilungen ----------------
    CONTAINER_TYPE_CHOICES = ["Standard", "High Cube", "Reefer", "Flat Rack", "Open Top"]
    CONTAINER_SIZES = [20.0, 40.0, 45.0]
    
    def random_container_id():
        # KORREKTUR: rng.choice mit korrekter Syntax
        alphabet = list("ABCDEFGHJKLMNPQRSTUVWXYZ23456789")
        random_chars = ''.join(rng.choice(alphabet, size=3))
        return f"CNT-{rng.integers(10000,99999)}-{random_chars}"

    # Container-Basisdaten mit Verteilungsunterstützung
    for r in reqs:
        col_name, field_type, dist_config, valueSource, customValues = r["name"], r["type"], r["dist_config"], r["valueSource"], r["customValues"]
        dist = (dist_config.distribution or "").strip().lower() if dist_config else ""

        # 1. UnitName - immer gleich (keine Verteilung nötig)
        if field_type == "unitname":
            df[col_name] = [random_container_id() for _ in range(rowCount)]

        # 2. Container Typ - mit kategorischer Verteilung
        elif field_type == "containertyp":
            if dist == "categorical":
                paramA = dist_config.parameterA if dist_config else None
                paramB = dist_config.parameterB if dist_config else None
                
                values = _split_list(paramA) or CONTAINER_TYPE_CHOICES
                weights = [float(w, 1.0) for w in _split_list(paramB or "")]
                
                if not weights or len(weights) != len(values):
                    weights = [1.0] * len(values)
                
                df[col_name] = _categorical_exact(values, np.array(weights), rowCount)
            else:
                if valueSource == "custom":
                    df[col_name] = customValues
                else:
                    df[col_name] = [rng.choice(CONTAINER_TYPE_CHOICES) for _ in range(rowCount)]

        # 3. Container Größe - mit uniform/kategorischer Verteilung
        elif field_type == "attributesize":
            if dist == "uniform":
                paramA = float(dist_config.parameterA) if dist_config else 20.0
                paramB = float(dist_config.parameterB) if dist_config else 45.0
                df[col_name] = rng.uniform(paramA, paramB, size=rowCount).round(1)
            elif dist == "categorical":
                paramA = dist_config.parameterA if dist_config else None
                paramB = dist_config.parameterB if dist_config else None
                
                values = _split_list(paramA) or [str(x) for x in CONTAINER_SIZES]
                values = [float(v, 20.0) for v in values]  # Convert to float
                weights = [float(w, 1.0) for w in _split_list(paramB or "")]
                
                if not weights or len(weights) != len(values):
                    weights = [1.0] * len(values)
                
                df[col_name] = _categorical_exact(values, np.array(weights), rowCount)
            else:
                if valueSource == "custom":
                    df[col_name] = customValues
                else:
                    df[col_name] = rng.choice(CONTAINER_SIZES, size=rowCount)

        # 4. Container Gewicht - mit uniform/normal Verteilung
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
                df[col_name] = pd.NA  # Wird später basierend auf Größe berechnet

        # 5. Container Status - kategorische Verteilung
        elif field_type == "attributestatus":
            if dist == "categorical":
                paramA = dist_config.parameterA if dist_config else None
                paramB = dist_config.parameterB if dist_config else None
                
                values = _split_list(paramA) or ["loaded", "empty"]
                weights = [float(w, 1.0) for w in _split_list(paramB or "")]
                
                if not weights or len(weights) != len(values):
                    weights = [1.0] * len(values)
                
                df[col_name] = _categorical_exact(values, np.array(weights), rowCount)
            else:
                if valueSource == "custom":
                    df[col_name] = customValues
                else:
                    df[col_name] = pd.NA  # Wird später berechnet

        # 6. Container Direction - kategorische Verteilung
        elif field_type == "attributedirection":
            if dist == "categorical":
                paramA = dist_config.parameterA if dist_config else None
                paramB = dist_config.parameterB if dist_config else None
                
                values = _split_list(paramA) or ["import", "export"]
                weights = [float(w, 1.0) for w in _split_list(paramB or "")]
                
                if not weights or len(weights) != len(values):
                    weights = [1.0] * len(values)
                
                df[col_name] = _categorical_exact(values, np.array(weights), rowCount)
            else:
                if valueSource == "custom":
                    df[col_name] = customValues
                else:
                    df[col_name] = [rng.choice(["import", "export", "transshipment"]) for _ in range(rowCount)]

        # 7. Time In - uniform Verteilung über Datumsbereich
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
                df[col_name] = pd.NA  # Wird später mit Carrier-Daten gefüllt

        # 8. Time Out - uniform Verteilung über Datumsbereich
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
                df[col_name] = pd.NA  # Wird später mit Carrier-Daten gefüllt

        # 9. Dwell Time - uniform/normal Verteilung
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
                df[col_name] = pd.NA  # Wird später berechnet

    # ---------------- Carrier-Daten Integration ----------------
    carrier_field_types = {
        "shipname", "linername", "service_route", "servicename",
        "eta", "etd", "loadteu", "dischargeteu", "length_m"
    }
    
    needs_carrier = any(r["type"] in carrier_field_types for r in reqs)
    needs_times = any(r["type"] in {"timein", "timeout", "dwelltime"} for r in reqs)

    if needs_carrier or needs_times:
        n_carriers = max(5, min(50, rowCount // 3))
        carr = generate_carrierData(n_carriers)
        if valueSource == "custom":
            carr["serviceRoute"] = customValues
        carr_map = carr.set_index("id").to_dict("index")
        carrier_ids = rng.choice(carr["id"].tolist(), size=rowCount, replace=True)

        def get_carrier_value(cid, field):
            return carr_map.get(cid, {}).get(field)

        # Carrier-Felder zuweisen
        carrier_mapping = {
            "shipname": "shipName",
            "linername": "linerName", 
            "service_route": "serviceRoute",
            "servicename": "serviceName", 
            "eta": "eta",
            "etd": "etd", 
            "loadteu": "loadTEU",
            "dischargeteu": "dischargeTEU",
            "length_m": "length_m"
        }
        
        for r in reqs:
            if r["type"] in carrier_mapping:
                df[r["name"]] = [get_carrier_value(cid, carrier_mapping[r["type"]]) for cid in carrier_ids]

        # Realistische Zeit-Abhängigkeiten anwenden
        _apply_time_dependencies(df, reqs, carrier_ids, carr_map, rng)

        # Realistische Abhängigkeiten anwenden
        _apply_container_dependencies(df, reqs, carrier_ids, carr_map, rng)

    # ---------------- Finale Abhängigkeiten berechnen ----------------
    _calculate_final_dependencies(df, reqs, rng)

    return df


def _apply_time_dependencies(df, reqs, carrier_ids, carr_map, rng):
    """
    Wendet realistische Zeit-Abhängigkeiten an.
    
    KORREKTE LOGIK:
    - TimeIn >= ETA (Container kommt NACH/beim Schiff an)
    - TimeOut <= ETD (Container geht VOR Schiff ab) 
    - TimeOut > TimeIn (Positive Verweilzeit)
    """
    
    def get_carrier_value(cid, field):
        return carr_map.get(cid, {}).get(field)
    
    timein_fields = [r for r in reqs if r["type"] == "timein" and pd.isna(df[r["name"]]).any()]
    timeout_fields = [r for r in reqs if r["type"] == "timeout" and pd.isna(df[r["name"]]).any()]
    
    if timein_fields or timeout_fields:
        # Carrier-Zeiten holen
        eta_times = pd.to_datetime([get_carrier_value(cid, "eta") for cid in carrier_ids])
        etd_times = pd.to_datetime([get_carrier_value(cid, "etd") for cid in carrier_ids])
        
        time_in, time_out = [], []
        
        for i, (eta, etd) in enumerate(zip(eta_times, etd_times)):
            # TimeIn: 0-36 Stunden NACH ETA
            max_hours_after_eta = min(36, int((etd - eta).total_seconds() / 3600) - 1)
            if max_hours_after_eta < 0:
                hours_after_eta = 0
            else:
                hours_after_eta = rng.integers(0, max_hours_after_eta + 1)
            
            t_in = eta + pd.Timedelta(hours=hours_after_eta)
            time_in.append(t_in)
            
            # TimeOut: 1-24 Stunden VOR ETD (und NACH TimeIn)
            max_hours_before_etd = min(24, int((etd - t_in).total_seconds() / 3600) - 1)
            if max_hours_before_etd < 1:
                hours_before_etd = 1
            else:
                hours_before_etd = rng.integers(1, max_hours_before_etd + 1)
            
            t_out = etd - pd.Timedelta(hours=hours_before_etd)
            
            # Sicherstellen dass timeOut > timeIn
            if t_out <= t_in:
                t_out = t_in + pd.Timedelta(hours=1)
            
            time_out.append(t_out)
        
        # Felder füllen
        for timein_field in timein_fields:
            df[timein_field["name"]] = [t.strftime("%Y-%m-%d %H:%M:%S") for t in time_in]
        
        for timeout_field in timeout_fields:
            df[timeout_field["name"]] = [t.strftime("%Y-%m-%d %H:%M:%S") for t in time_out]


def _apply_container_dependencies(df, reqs, carrier_ids, carr_map, rng):
    """Wendet realistische Container-Carrier Abhängigkeiten an"""
    
    def get_carrier_value(cid, field):
        return carr_map.get(cid, {}).get(field)
    
    # Größe basierend auf Schiffskapazität (falls nicht schon durch Verteilung gesetzt)
    size_fields = [r for r in reqs if r["type"] == "attributesize" and pd.isna(df[r["name"]]).any()]
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

    # Typ basierend auf Route (falls nicht schon durch Verteilung gesetzt)
    type_fields = [r for r in reqs if r["type"] == "containertyp" and pd.isna(df[r["name"]]).any()]
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
    """Berechnet finale Abhängigkeiten zwischen Container-Feldern"""
    
    # Gewicht basierend auf Größe und Typ (falls nicht schon durch Verteilung gesetzt)
    weight_fields = [r for r in reqs if r["type"] == "attributeweight" and pd.isna(df[r["name"]]).any()]
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
                else:  # 45ft
                    base = rng.integers(4800, 5200)
                
                if "reefer" in str(container_type).lower():
                    base += rng.integers(500, 1000)
                elif "flat" in str(container_type).lower() or "open" in str(container_type).lower():
                    base -= rng.integers(200, 500)
                
                weights.append(base)
            
            df[field["name"]] = weights

    # Status basierend auf Gewicht (falls nicht schon durch Verteilung gesetzt)
    status_fields = [r for r in reqs if r["type"] == "attributestatus" and pd.isna(df[r["name"]]).any()]
    if status_fields:
        for field in status_fields:
            weight_col = next((r["name"] for r in reqs if r["type"] == "attributeweight"), None)
            if weight_col:
                df[field["name"]] = ["empty" if w < 2500 else "loaded" for w in df[weight_col]]
            else:
                df[field["name"]] = [rng.choice(["loaded", "empty"], p=[0.7, 0.3]) for _ in range(len(df))]

    # DwellTime berechnen aus TimeIn/TimeOut (falls nicht schon durch Verteilung gesetzt)
    dwell_fields = [r for r in reqs if r["type"] == "dwelltime" and pd.isna(df[r["name"]]).any()]
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
                            dwell_times.append(rng.uniform(6.0, 72.0).round(2))
                    except:
                        dwell_times.append(rng.uniform(6.0, 72.0).round(2))
                df[field["name"]] = dwell_times
            else:
                df[field["name"]] = [rng.uniform(6.0, 72.0).round(2) for _ in range(len(df))]

rng = np.random.default_rng()
def _categorical_exact(values: list, weights: np.ndarray, size: int) -> list:
    """Gibt eine Liste der Länge `size` zurück, deren Elemente entsprechend den Gewichten verteilt sind.
    Es wird deterministisch gerundet: zunächst floor(weights*size) und verbleibende Elemente
    nach dem größten Anteil (Methode des größten Restes) verteilt. Das Ergebnis wird
    anschließend durchmischt.
    """
    if len(values) == 0:
        return [None] * size
    if weights is None or len(weights) != len(values):
        return [random.choice(values) for _ in range(size)]

    # normalisieren
    w = np.array(weights, dtype=float)
    if w.sum() <= 0:
        w = np.ones_like(w)
    w = w / w.sum()

    # exakte Zählungen mittels Abrunden (floor) + Verteilung der Restplätze nach größtem Anteil
    target = w * size
    counts = np.floor(target).astype(int)
    remainder = size - counts.sum()
    if remainder > 0:
        frac = target - np.floor(target)
        # Indizes nach absteigendem Nachkommaanteil sortieren
        idxs = np.argsort(-frac)
        for i in range(remainder):
            counts[idxs[i % len(idxs)]] += 1

    out = []
    for val, c in zip(values, counts):
        out.extend([val] * int(c))
    # Falls durch Rundung die Länge abweicht, anpassen
    if len(out) < size:
        out.extend([values[0]] * (size - len(out)))
    elif len(out) > size:
        out = out[:size]

    # Ergebnis deterministisch durch den RNG mischen
    arr = np.array(out, dtype=object)
    rng.shuffle(arr)
    return arr.tolist()

def _split_list(s: Optional[str]) -> list[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]