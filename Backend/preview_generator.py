from typing import List, Optional
from mimesis import Person, Gender as MimesisGender  # mimesis statt faker
from mimesis.enums import Locale
from scipy.stats import norm, uniform, gamma, poisson, binom
import random
import pandas as pd
import numpy as np
from field_schemas import FrontendField, DistributionConfig

# mimesis Generatoren für verschiedene Sprachen
person_generators = {
    'german': Person(Locale.DE),
    'english': Person(Locale.EN),
    'french': Person(Locale.FR),
    'spanish': Person(Locale.ES),
    'turkish': Person(Locale.TR),
    'russian': Person(Locale.RU),
    'chinese': Person(Locale.ZH),
    'japanese': Person(Locale.JA),
    'italian': Person(Locale.IT)
}

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


def _get_name_generator(nationality: str = 'german'):
    """Holt den passenden Namen-Generator basierend auf Nationalität"""
    return person_generators.get(nationality, person_generators['german'])


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
        
        # Nationalität aus dem Feld extrahieren (falls vorhanden)
        nationality = getattr(field, 'nationality', 'german')

        if ftype in ["name", "vorname", "nachname", "vollständigername"]:
            paramA = dist_config.parameterA if dist_config else None
            paramB = dist_config.parameterB if dist_config else None
            person_gen = _get_name_generator(nationality)

            if dist == "categorical":
                values = _split_list(paramA) or [person_gen.full_name() for _ in range(5)]
                weights = [_to_float(w, 1.0) for w in _split_list(paramB or "")]
                if not weights or len(weights) != len(values):
                    weights = [1.0] * len(values)
                weights = np.array(weights, dtype=float)
                weights = weights / weights.sum()
                data = np.random.choice(values, size=num_rows, p=weights).tolist()
            else:
                # Standard: Realistische Namen basierend auf Nationalität
                data = [person_gen.full_name() for _ in range(num_rows)]

        elif ftype in ["geschlecht", "gender"]:
            paramA = dist_config.parameterA if dist_config else None
            paramB = dist_config.parameterB if dist_config else None
            person_gen = _get_name_generator(nationality)

            if dist == "categorical":
                values = _split_list(paramA) or ["M", "W", "D"]
                weights = [_to_float(w, 1.0) for w in _split_list(paramB or "")]
                if not weights or len(weights) != len(values):
                    weights = [1.0] * len(values)
                weights = np.array(weights, dtype=float)
                weights = weights / weights.sum()
                data = np.random.choice(values, size=num_rows, p=weights).tolist()
            else:
                # Realistische Geschlechter mit mimesis
                data = []
                for _ in range(num_rows):
                    gender = person_gen.gender()
                    # Anpassung an deutsche Bezeichnungen falls nötig
                    if nationality == 'german':
                        if gender == 'Male':
                            data.append('M')
                        elif gender == 'Female':
                            data.append('W')
                        else:
                            data.append('D')
                    else:
                        data.append(gender[0])  # Erster Buchstabe

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

        elif ftype in ["integer", "alter", "plz", "hausnummer"]:
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
                data = [random.randint(100, 9999) for _ in range(num_rows)]
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

        elif ftype in ["adresse", "straße", "stadt", "land"]:
            person_gen = _get_name_generator(nationality)
            if ftype == "adresse":
                data = [person_gen.address() for _ in range(num_rows)]
            elif ftype == "straße":
                data = [person_gen.street_name() for _ in range(num_rows)]
            elif ftype == "stadt":
                data = [person_gen.city() for _ in range(num_rows)]
            elif ftype == "land":
                data = [person_gen.country() for _ in range(num_rows)]
            else:
                data = [f"{field.name}_{i}" for i in range(num_rows)]

        elif ftype in ["email", "e-mail"]:
            person_gen = _get_name_generator(nationality)
            data = [person_gen.email() for _ in range(num_rows)]

        elif ftype in ["telefon", "handynummer"]:
            person_gen = _get_name_generator(nationality)
            data = [person_gen.telephone() for _ in range(num_rows)]

        else:
            data = [f"{field.name}_{i}" for i in range(num_rows)]

        columns[field.name] = data
    
    # 2) Abhängige Felder verarbeiten (Dependency nutzen)
    for field in dependent_fields:
        dep = (field.dependency or "").strip()
        nationality = getattr(field, 'nationality', 'german')
        
        if not dep:
            columns[field.name] = [f"{field.name}_{i}" for i in range(num_rows)]
            continue

        if dep not in columns:
            columns[field.name] = [None] * num_rows
            continue
        
        dep_values = columns[dep]
        ftype = (field.type or "").strip().lower()

        if ftype in ["name", "vorname"]:
            person_gen = _get_name_generator(nationality)
            data = []
            for gv in dep_values:
                g = (gv or "").strip().lower()
                if g in ["m", "male", "männlich"]:
                    first = person_gen.first_name(gender=MimesisGender.MALE)
                elif g in ["w", "f", "female", "weiblich"]:
                    first = person_gen.first_name(gender=MimesisGender.FEMALE)
                else:
                    first = person_gen.first_name()
                
                if ftype == "vorname":
                    data.append(first)
                else:
                    last = person_gen.last_name()
                    data.append(f"{first} {last}")
            columns[field.name] = data
        else:
            columns[field.name] = list(dep_values)

    return pd.DataFrame(columns)