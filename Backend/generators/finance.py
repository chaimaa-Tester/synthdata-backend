from typing import List, Optional
from mimesis import *
from mimesis.enums import *
import random
import pandas as pd
import numpy as np
from field_schemas import FrontendField, DistributionConfig

TRANSACTION_TYPES: List[str] = [
    "SEPA-Überweisung",
    "Gehalt / Lohn",
    "Karten-Zahlung (Debit)",
    "Gebühren / Kontoführungsgebühr",
    "Rückerstattung / Refund",
    "Internationale Überweisung (Swift)",
    "Online-Zahlung",
    "Mobile Payment",
    "Abonnement / Abo-Zahlung"
]

person_gen = Person(Locale.DE)

rng = np.random.default_rng()

def generate_financeData(fields: List[FrontendField], num_rows: int, as_text_for_sheets: bool = False) -> pd.DataFrame:
    columns: dict[str, list] = {}

    # Felder trennen, erst unabhängig dann abhängig
    independent_fields = [f for f in fields if not (f.dependency or "").strip()]
    dependent_fields = [f for f in fields if (f.dependency or "").strip()]

    # 1. Unabhängige Felder generieren
    for field in independent_fields:
        ftype = (field.type or "").strip()
        dist_config = field.distributionConfig
        dist = (dist_config.distribution or "").strip().lower() if dist_config else ""

        if ftype in ["betrag"]:
            paramA = _to_float(dist_config.parameterA) if dist_config else None
            paramB = _to_float(dist_config.parameterB) if dist_config else None

            if dist == "uniform":
                low = paramA if paramA is not None else 0
                high = paramB if paramB is not None else 999999999
                if high <= low:
                    high = low + 1.0
                arr = rng.uniform(low, high, size=num_rows)
                arr = np.round(np.asarray(arr, dtype=float), 2)
                data = _maybe_as_text(arr.tolist(), as_text_for_sheets)

            elif dist == "normal":
                mean = paramA if paramA is not None else 175.0
                sd = paramB if paramB is not None else 10.0
                arr = rng.normal(loc=mean, scale=sd, size=num_rows)
                arr = np.clip(arr, 100.0, 220.0)
                arr = np.round(arr.astype(float), 1)
                data = _maybe_as_text(arr.tolist(), as_text_for_sheets)
            
            else:
                data = [round(random.uniform(1, 1000), 2) for _ in range(num_rows)]
        
        elif ftype in ["kontonummer"]:
            paramA = _to_float(dist_config.parameterA) if dist_config else None
            paramB = _to_float(dist_config.parameterB) if dist_config else None

            # if dist == "uniform":
            #     low = int(paramA or 0)
            #     high = int(paramB or 100)
            #     if high < low:
            #         high = low
            #     data = [random.randint(low, high) for _ in range(num_rows)]
            #     data = _maybe_as_text(data, as_text_for_sheets)
            data = [gen_german_account() for _ in range(num_rows)]
        
        elif ftype in ["transaktionsdatum"]:
            paramA = (dist_config.parameterA if dist_config else None) or "2010-01-01"
            paramB = (dist_config.parameterB if dist_config else None) or "2025-12-31"
            start = pd.to_datetime(paramA).date()
            end = pd.to_datetime(paramB).date()
            if end < start:
                start, end = end, start
            start_ord = start.toordinal()
            end_ord = end.toordinal()
            ordinals = rng.integers(start_ord, end_ord + 1, size=num_rows)
            data = [pd.Timestamp.fromordinal(int(o)).date().isoformat() for o in ordinals]

        elif ftype in ["transaktionsart"]:
            data = [random.choice(TRANSACTION_TYPES) for _ in range(num_rows)]

        else:
            data = [f"{field.name}_{i}" for i in range(num_rows)]   

        columns[field.name] = data

    # 2. Abhängige Felder generieren
    for field in dependent_fields:
        dep_raw = (field.dependency or "").strip()

        if not dep_raw:
            columns[field.name] = [f"{field.name}_{i}" for i in range(num_rows)]
            continue

        dep_key = None
        if dep_raw in columns:
            dep_key = dep_raw
        else:
            for k in columns.keys():
                if k.lower() == dep_raw.lower():
                    dep_key = k
                    break
        if not dep_key:
            # Versuche comma-separierte Einträge und finde das erste verfügbare Matching
            for candidate in [d.strip() for d in dep_raw.split(",") if d.strip()]:
                for k in columns.keys():
                    if k.lower() == candidate.lower():
                        dep_key = k
                        break
                if dep_key:
                    break

        if not dep_key:
            # Warnung ausgeben wenn nicht beim Feldtyp eingibt und None-Werte zuweisen
            print(f"WARNUNG: Dependency '{dep_raw}' für Feld '{field.name}' nicht gefunden. Verfügbare Spalten: {list(columns.keys())}")
            columns[field.name] = [None] * num_rows
            continue

        dep_values = columns[dep_key]
        ftype = (field.type or "").strip()

        if ftype in ["betrag"]:
            amount_ranges = {
                "SEPA-Überweisung": (10, 5000),
                "Gehalt / Lohn": (1000, 5000),
                "Karten-Zahlung (Debit)": (5, 200),
                "Gebühren / Kontoführungsgebühr": (1, 50),
                "Rückerstattung / Refund": (5, 2000),
                "Internationale Überweisung (Swift)": (50, 10000),
                "Online-Zahlung": (5, 500),
                "Mobile Payment": (1, 300),
                "Abonnement / Abo-Zahlung": (5, 50)
            }
            data = []
            for t_type in dep_values:
                low, high = amount_ranges.get(t_type, 69)
                data.append(round(random.uniform(low, high), 2))
        
            columns[field.name] = data
        else:
            columns[field.name] = list(dep_values)
    
    return pd.DataFrame(columns)




def _to_float(value, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
           
def _split_list(s: Optional[str]) -> list[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

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

def _maybe_as_text(seq, as_text: bool) -> list:
    if not as_text:
        return list(seq)
    out = []
    for v in seq:
        if isinstance(v, (int, float, np.integer, np.floating)):
            # Format mit deutschem Dezimaltrennzeichen; behalte führende 0 (z.B. 0,50)
            val_str = format(v, '.2f').replace('.', ',')
            out.append(f"'{val_str}")
        else:
            out.append(v)
    return out

def gen_german_account():
    """
    Generiert:
    - kontonummer: 10-stellige, linksgepadete Nummer (String) + int
    """
    # realistisch: Kontonummer bis 10 Stellen, oft mit führenden Nullen
    k_num = "".join(str(random.randint(0,9)) for _ in range(10))
    return k_num