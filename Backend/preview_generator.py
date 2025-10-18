from typing import List, Optional
from faker import Faker
from scipy.stats import norm, uniform, gamma, poisson, binom
import random
import pandas as pd
import numpy as np
from field_schemas import FrontendField, DistributionConfig

fake = Faker()
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

        if ftype in ["name"]:
            paramA = dist_config.parameterA if dist_config else None
            paramB = dist_config.parameterB if dist_config else None

            if dist == "categorical":
                values = _split_list(paramA) or [fake.name() for _ in range(5)]
                weights = [_to_float(w, 1.0) for w in _split_list(paramB or "")]
                if not weights or len(weights) != len(values):
                    weights = [1.0] * len(values)
                weights = np.array(weights, dtype=float)
                weights = weights / weights.sum()
                data = np.random.choice(values, size=num_rows, p=weights).tolist()
            else:
                data = [fake.name() for _ in range(num_rows)]

        elif ftype in ["geschlecht"]:
            paramA = dist_config.parameterA if dist_config else None
            paramB = dist_config.parameterB if dist_config else None

            if dist == "categorical":
                values = _split_list(paramA) or [fake.passport_gender() for _ in range(5)]
                weights = [_to_float(w, 1.0) for w in _split_list(paramB or "")]
                if not weights or len(weights) != len(values):
                    weights = [1.0] * len(values)
                weights = np.array(weights, dtype=float)
                weights = weights / weights.sum()
                data = np.random.choice(values, size=num_rows, p=weights).tolist()
            else:
                data = [fake.passport_gender() for _ in range(num_rows)]

        elif ftype in ["körpergröße", "float"]:
            paramA = _to_float(dist_config.parameterA) if dist_config else None
            paramB = _to_float(dist_config.parameterB) if dist_config else None


            # if dist == "normal":
            #     mu = paramA or 0.0
            #     sigma = paramB if (paramB and paramB > 0) else 1.0
            #     arr = norm.rvs(loc=mu, scale=sigma, size=num_rows)

            if dist == "uniform":
                low = paramA if paramA is not None else 0.0
                high = paramB if paramB is not None else 100.0
                if high <= low:
                    high = low + 1.0
                arr = rng.uniform(low, high, size = num_rows)

            # elif dist == "gamma":
            #     shape = paramA or 1
            #     scale = paramB or 1
            #     arr = gamma.rvs(a=shape, scale=scale, size=num_rows)
            # elif dist == "lognormal":
            #     mu = paramA or 0
            #     sigma = paramB or 1
            #     arr = np.random.lognormal(mean=mu, sigma=sigma, size=num_rows)
            # elif dist == "exponential":
            #     rate = paramA or 1
            #     arr = np.random.exponential(scale=1 / rate, size=num_rows)
            else:
                arr = rng.uniform(low = 0.0, high = 100.0, size = num_rows)

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
            # elif dist == "normal":
            #     mu = paramA or 0
            #     sigma = paramB or 1
            #     # Erzeuge normale Verteilung und runde auf ganze Zahlen
            #     data = np.round(norm.rvs(loc=mu, scale=sigma, size=num_rows)).astype(int)
            #     data = _maybe_as_text(data.tolist(), as_text_for_sheets)
            # elif dist == "binomial": #'n' muss ≥ 0 und 'p' zwischen 0 und 1 sein.
            #     n = int(paramA) if paramA is not None else 10
            #     p = paramB or 0.5
            #     if n < 0:
            #         raise ValueError("'n' muss ≥ 0 sein.")
            #     if not (0 <= p <= 1):
            #         raise ValueError("'p' muss zwischen 0 und 1 sein.")
            #     data = binom.rvs(n=n, p=p, size=num_rows)
            #     data = _maybe_as_text(data.tolist(), as_text_for_sheets)
            # elif dist == "poisson":
            #     lam = float(paramA or 1)
            #     data = poisson.rvs(mu=lam, size=num_rows).tolist()
            #     data = _maybe_as_text(data.tolist(), as_text_for_sheets)
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

        else:
            data = [f"{field.name}_{i}" for i in range(num_rows)]

        columns[field.name] = data
    
    # 2) Abhängige Felder verarbeiten (Dependency nutzen)
    for field in dependent_fields:
        dep = (field.dependency or "").strip()
        # wenn Dependency fehlt oder leer, Fallback wie bisher
        if not dep:
            columns[field.name] = [f"{field.name}_{i}" for i in range(num_rows)]
            continue

        if dep not in columns:
            # Dependency nicht gefunden => None-Füllung als Signal
            columns[field.name] = [None] * num_rows
            continue
        
        dep_values = columns[dep]
        ftype = (field.type or "").strip().lower()

        if ftype in ["name"]:
            # Erzeuge Namen passend zum Geschlecht in dep_values
            data = []
            for gv in dep_values:
                g = (gv or "").strip().lower()
                if g.startswith("m"):
                    first = fake.first_name_male()
                elif g.startswith("w"):
                    first = fake.first_name_female()
                else:
                    first = fake.first_name()
                last = fake.last_name()
                data.append(f"{first} {last}")
            columns[field.name] = data
        else:
            # Default: Werte aus dem referenzierten Feld übernehmen
            columns[field.name] = list(dep_values)

    return pd.DataFrame(columns)
