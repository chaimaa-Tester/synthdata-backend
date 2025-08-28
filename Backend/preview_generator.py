from typing import List
from faker import Faker
from scipy.stats import norm, uniform, gamma
import random
import pandas as pd
import numpy as np
from field_schemas import FrontendField, DistributionConfig

fake = Faker()
rng = np.random.default_rng()


def _to_int(value, default=None):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _maybe_as_text(seq, as_text: bool) -> list:
    """
    Falls as_text=True: Zahlen als Text mit führendem Apostroph ausgeben,
    um Google Sheets' automatische Datums-/Formatumwandlung zu verhindern.
    """
    if not as_text:
        return list(seq)
    out = []
    for v in seq:
        if isinstance(v, (int, float, np.integer, np.floating)):
            out.append(f"'{v}")  # Apostroph schützt vor Datumskonvertierung
        else:
            out.append(v)
    return out


def generate_dummy_data(fields: List[FrontendField], num_rows: int, as_text_for_sheets: bool = False) -> pd.DataFrame:
    columns: dict[str, list] = {}

    for field in fields:
        ftype = field.type.strip().lower() if field.type else ""
        dist_config = field.distributionConfig
        dist = dist_config.distribution.lower() if (dist_config and dist_config.distribution) else None

        # ---------- TEXTUAL TYPES ----------
        if ftype in ("string", "text"):
            data = [fake.word() for _ in range(num_rows)]

        # ---------- INTEGER TYPES ----------
        elif ftype in ("int", "integer"):
            paramA = _to_float(dist_config.parameterA) if dist_config else None
            paramB = _to_float(dist_config.parameterB) if dist_config else None
            low = _to_int(paramA, 0)
            high = _to_int(paramB, 100)
            if high is None:
                high = 100
            if low is None:
                low = 0
            if high < low:
                low, high = high, low
            data = rng.integers(low, high + 1, size=num_rows).tolist()
            data = _maybe_as_text(data, as_text_for_sheets)

        # ---------- FLOAT/DOUBLE TYPES ----------
        elif ftype in ("float", "double", "number", "numeric"):
            paramA = _to_float(dist_config.parameterA) if dist_config else None
            paramB = _to_float(dist_config.parameterB) if dist_config else None

            if dist == "normal":
                mu = paramA if paramA is not None else 0.0
                sigma = paramB if (paramB is not None and paramB > 0) else 1.0
                arr = norm.rvs(loc=mu, scale=sigma, size=num_rows)
            elif dist == "uniform":
                low = paramA if paramA is not None else 0.0
                high = paramB if paramB is not None else 100.0
                if high <= low:
                    high = low + 1.0
                arr = uniform.rvs(loc=low, scale=high - low, size=num_rows)
            elif dist == "gamma":
                shape = paramA if (paramA is not None and paramA > 0) else 1.0
                scale = paramB if (paramB is not None and paramB > 0) else 1.0
                arr = gamma.rvs(a=shape, scale=scale, size=num_rows)
            else:
                arr = uniform.rvs(loc=0.0, scale=1000.0, size=num_rows)

            arr = np.round(np.asarray(arr, dtype=float), 2)
            data = _maybe_as_text(arr.tolist(), as_text_for_sheets)

        # ---------- DATE TYPES ----------
        elif ftype == "date":
            paramA = (dist_config.parameterA) if dist_config and dist_config.parameterA else None
            paramB = (dist_config.parameterB) if dist_config and dist_config.parameterB else None
            if dist == "uniform":
                start = pd.to_datetime(paramA or "2000-01-01").date()
                end = pd.to_datetime(paramB or "2025-01-01").date()
                if end < start:
                    start, end = end, start
                start_ord = start.toordinal()
                end_ord = end.toordinal()
                ordinals = rng.integers(start_ord, end_ord + 1, size=num_rows)
                data = [pd.Timestamp.fromordinal(int(o)).date().isoformat() for o in ordinals]
            else:
                data = [fake.date() for _ in range(num_rows)]

        # ---------- DATETIME/TIME/ID/EMAIL/NAME ----------
        elif ftype in ("datetime",):
            data = [fake.iso8601() for _ in range(num_rows)]
        elif ftype in ("time",):
            data = [fake.time() for _ in range(num_rows)]
        elif ftype in ("uuid",):
            data = [fake.uuid4() for _ in range(num_rows)]
        elif ftype in ("email",):
            data = [fake.email() for _ in range(num_rows)]
        elif ftype in ("name",):
            data = [fake.name() for _ in range(num_rows)]

        # ---------- FALLBACK ----------
        else:
            data = [f"{field.name}_{i}" for i in range(num_rows)]

        columns[field.name] = data

    return pd.DataFrame(columns)
