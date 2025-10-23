from typing import List, Optional
from mimesis import *  # mimesis statt faker
from mimesis.enums import *
from scipy.stats import norm, uniform, gamma, poisson, binom
import random
import pandas as pd
import numpy as np
from field_schemas import FrontendField, DistributionConfig

# Deutscher mimesis Generator für bessere Namen und Geschlechts-Konsistenz
locale = Locale.DE
person_gen = Person(locale)

rng = np.random.default_rng()


def _to_int(value, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_float(value, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _split_list(s: Optional[str]) -> list[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _maybe_as_text(seq, as_text: bool) -> list:
    if not as_text:
        return list(seq)
    out = []
    for v in seq:
        if isinstance(v, (int, float, np.integer, np.floating)):
            val_str = format(v, '.2f').replace('.', ',').lstrip("0")
            out.append(f"'{val_str}")
        else:
            out.append(v)
    return out


def generate_dummy_data(fields: List[FrontendField], num_rows: int, as_text_for_sheets: bool = False) -> pd.DataFrame:
    columns: dict[str, list] = {}

    # Felder trennen: erst unabhängig, dann abhängig
    independent_fields = [f for f in fields if not (f.dependency or "").strip()]
    dependent_fields = [f for f in fields if (f.dependency or "").strip()]

    # 1) Unabhängige Felder generieren
    for field in independent_fields:
        ftype = (field.type or "").strip().lower()
        dist_config = field.distributionConfig
        dist = (dist_config.distribution or "").strip().lower() if dist_config else ""

        if ftype in ["name", "vorname", "nachname"]:
            paramA = dist_config.parameterA if dist_config else None
            paramB = dist_config.parameterB if dist_config else None

            if dist == "categorical":
                if ftype == "vorname":
                    values = _split_list(paramA) or [person_gen.first_name() for _ in range(5)]
                elif ftype == "nachname":
                    values = _split_list(paramA) or [person_gen.last_name() for _ in range(5)]
                else:  # "name" = vollständiger Name
                    values = _split_list(paramA) or [person_gen.full_name() for _ in range(5)]
                
                weights = [_to_float(w, 1.0) for w in _split_list(paramB or "")]
                if not weights or len(weights) != len(values):
                    weights = [1.0] * len(values)
                weights = np.array(weights, dtype=float)
                weights = weights / weights.sum()
                data = np.random.choice(values, size=num_rows, p=weights).tolist()
            else:
                # Standard: Realistische Namen je nach Typ
                if ftype == "vorname":
                    data = [person_gen.first_name() for _ in range(num_rows)]
                elif ftype == "nachname":
                    data = [person_gen.last_name() for _ in range(num_rows)]
                else:  # "name" = vollständiger Name
                    data = [person_gen.full_name() for _ in range(num_rows)]

        elif ftype in ["geschlecht", "gender"]:
            paramA = dist_config.parameterA if dist_config else None
            paramB = dist_config.parameterB if dist_config else None

            if dist == "categorical":
                values = _split_list(paramA) or person_gen.gender()
                weights = [_to_float(w, 1.0) for w in _split_list(paramB or "")]
                if not weights or len(weights) != len(values):
                    weights = [1.0] * len(values)
                weights = np.array(weights, dtype=float)
                weights = weights / weights.sum()
                data = np.random.choice(values, size=num_rows, p=weights).tolist()
            else:
                # BESTE LÖSUNG: Direkte 50/50 Verteilung M/W - zuverlässig und logisch
                data = [person_gen.gender() for _ in range(num_rows)]

        elif ftype in ["körpergröße", "float", "gewicht"]:
            paramA = _to_float(dist_config.parameterA) if dist_config else None
            paramB = _to_float(dist_config.parameterB) if dist_config else None

            if dist == "uniform":
                low = paramA if paramA is not None else 0.0
                high = paramB if paramB is not None else 100.0
                if high <= low:
                    high = low + 1.0
                arr = rng.uniform(low, high, size=num_rows)

            else:
                arr = rng.uniform(low=0.0, high=100.0, size=num_rows)

            arr = np.round(np.asarray(arr, dtype=float), 2)
            data = _maybe_as_text(arr.tolist(), as_text_for_sheets)

        elif ftype in ["integer", "alter"]:
            paramA = _to_float(dist_config.parameterA) if dist_config else None
            paramB = _to_float(dist_config.parameterB) if dist_config else None

            if dist == "uniform":
                low = int(paramA or 0)
                high = int(paramB or 100)
                if high < low:
                    high = low
                data = [random.randint(low, high) for _ in range(num_rows)]
                data = _maybe_as_text(data, as_text_for_sheets)
            else:
                data = [random.randint(0, 9999) for _ in range(num_rows)]
                data = _maybe_as_text(data, as_text_for_sheets)
                
        
        elif ftype == "date":
            paramA = (dist_config.parameterA if dist_config else None) or "2000-01-01"
            paramB = (dist_config.parameterB if dist_config else None) or "2025-01-01"
            start = pd.to_datetime(paramA).date()
            end = pd.to_datetime(paramB).date()
            if end < start:
                start, end = end, start
            start_ord = start.toordinal()
            end_ord = end.toordinal()
            ordinals = rng.integers(start_ord, end_ord + 1, size=num_rows)
            data = [pd.Timestamp.fromordinal(int(o)).date().isoformat() for o in ordinals]

        elif ftype in ["straße", "stadt", "land", "plz", "hausnummer"]:
            if ftype == "straße":
                data = [Address(locale).street_name() for _ in range(num_rows)]
            elif ftype == "stadt":
                data = [Address(locale).city() for _ in range(num_rows)]
            elif ftype == "land":
                data = [Address(locale).country() for _ in range(num_rows)]
            elif ftype == "plz":
                data = [Address(locale).postal_code() for _ in range(num_rows)]
            elif ftype == "hausnummer":
                data = [Address(locale).street_number() for _ in range (num_rows)]
            else:
                data = [f"{field.name}_{i}" for i in range(num_rows)]

        elif ftype in ["email", "e-mail"]:
            data = [person_gen.email() for _ in range(num_rows)]

        elif ftype in ["telefon", "handynummer"]:
            data = [person_gen.telephone() for _ in range(num_rows)]

        else:
            data = [f"{field.name}_{i}" for i in range(num_rows)]

        columns[field.name] = data
    
    # 2) Abhängige Felder verarbeiten (Dependency nutzen)
    for field in dependent_fields:
        dep = (field.dependency or "").strip()
        
        if not dep:
            columns[field.name] = [f"{field.name}_{i}" for i in range(num_rows)]
            continue

        if dep not in columns:
            columns[field.name] = [None] * num_rows
            continue
        
        dep_values = columns[dep]
        ftype = (field.type or "").strip().lower()

        if ftype in ["name", "vorname", "nachname"]:
            data = []
            for gv in dep_values:
                # g = (gv or "").strip().upper()  # GROSSBUCHSTABEN für bessere Erkennung
                
                if gv == "Männlich":  # Nur exakte M Erkennung
                    if ftype == "vorname":
                        data.append(person_gen.first_name(gender=Gender.MALE))
                    elif ftype == "nachname":
                        data.append(person_gen.last_name())
                    else:  # "name" = vollständiger Name
                        first = person_gen.first_name(gender=Gender.MALE)
                        last = person_gen.last_name()
                        data.append(f"{first} {last}")
                        
                elif gv == "Weiblich":  # Nur exakte W Erkennung
                    if ftype == "vorname":
                        data.append(person_gen.first_name(gender=Gender.FEMALE))
                    elif ftype == "nachname":
                        data.append(person_gen.last_name())
                    else:  # "name" = vollständiger Name
                        first = person_gen.first_name(gender=Gender.FEMALE)
                        last = person_gen.last_name()
                        data.append(f"{first} {last}")
                        
                else:
                    # FALLBACK: Bei unbekannten Werten -> Default WEIBLICH (oder Fehler-Behandlung)
                    print(f"WARNUNG: Unbekanntes Geschlecht '{gv}', verwende weiblich als Fallback")
                    if ftype == "vorname":
                        data.append(person_gen.first_name())
                    elif ftype == "nachname":
                        data.append(person_gen.last_name())
                    else:  # "name" = vollständiger Name
                        first = person_gen.first_name()
                        data.append(f"{first} {last}")
                        
            columns[field.name] = data
        else:
            columns[field.name] = list(dep_values)

    return pd.DataFrame(columns)