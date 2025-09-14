from typing import List
from faker import Faker
from scipy.stats import norm, uniform, gamma, poisson, binom
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


def _split_list(s: str) -> list[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _maybe_as_text(seq, as_text: bool) -> list:
    if not as_text:
        return list(seq)
    out = []
    for v in seq:
        if isinstance(v, (int, float, np.integer, np.floating)):
            out.append(f"'{v}")
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

        if ftype in ["string", "text"]:
            paramA = dist_config.parameterA if dist_config else None
            paramB = dist_config.parameterB if dist_config else None

            if dist == "uniform":
                values = _split_list(paramA) or [fake.word() for _ in range(5)]
                data = [random.choice(values) for _ in range(num_rows)]
            elif dist == "categorical":
                values = _split_list(paramA) or [fake.word() for _ in range(5)]
                weights = [_to_float(w, 1.0) for w in _split_list(paramB or "")]
                if not weights or len(weights) != len(values):
                    weights = [1.0] * len(values)
                weights = np.array(weights, dtype=float)
                weights = weights / weights.sum()
                data = np.random.choice(values, size=num_rows, p=weights).tolist()
            else:
                data = [fake.word() for _ in range(num_rows)]

        elif ftype in ("float", "double", "number", "numeric"):
            paramA = _to_float(dist_config.parameterA) if dist_config else None
            paramB = _to_float(dist_config.parameterB) if dist_config else None

            if dist == "normal":
                mu = paramA or 0.0
                sigma = paramB if (paramB and paramB > 0) else 1.0
                arr = norm.rvs(loc=mu, scale=sigma, size=num_rows)
            elif dist == "uniform":
                low = paramA if paramA is not None else 0.0
                high = paramB if paramB is not None else 100.0
                if high <= low:
                    high = low + 1.0
                arr = uniform.rvs(loc=low, scale=high - low, size=num_rows)
            elif dist == "gamma":
                shape = paramA or 1
                scale = paramB or 1
                arr = gamma.rvs(a=shape, scale=scale, size=num_rows)
            elif dist == "lognormal":
                mu = paramA or 0
                sigma = paramB or 1
                arr = np.random.lognormal(mean=mu, sigma=sigma, size=num_rows)
            elif dist == "exponential":
                rate = paramA or 1
                arr = np.random.exponential(scale=1 / rate, size=num_rows)
            else:
                arr = uniform.rvs(loc=0.0, scale=1000.0, size=num_rows)

            arr = np.round(np.asarray(arr, dtype=float), 2)
            data = _maybe_as_text(arr.tolist(), as_text_for_sheets)

        elif ftype in ["int", "integer"]:
            paramA = _to_float(dist_config.parameterA) if dist_config else None
            paramB = _to_float(dist_config.parameterB) if dist_config else None

            if dist == "uniform":
                low = int(paramA or 0)
                high = int(paramB or 100)
                if high < low:
                    high = low
                data = [random.randint(low, high) for _ in range(num_rows)]
            elif dist == "normal":
                mu = paramA or 0
                sigma = paramB or 1
                data = np.round(norm.rvs(loc=mu, scale=sigma, size=num_rows)).astype(int).tolist()
            elif dist == "binomial":
                n = int(paramA or 10)
                p = float(paramB or 0.5)
                data = binom.rvs(n=n, p=p, size=num_rows).tolist()
            elif dist == "poisson":
                lam = float(paramA or 1)
                data = poisson.rvs(mu=lam, size=num_rows).tolist()
            else:
                data = [random.randint(100, 9999) for _ in range(num_rows)]

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

    # 2) Abhängige Felder verarbeiten (neu)
    for field in dependent_fields:
        dep_fields = _split_list(field.dependency or "")
        ftype = (field.type or "").strip().lower()
        cfg = field.distributionConfig or DistributionConfig(distribution="", parameterA="", parameterB="", extraParams=[])
        dist = (cfg.distribution or "").strip().lower()

        # Dependencies vorhanden?
        if not all(d in columns for d in dep_fields):
            columns[field.name] = [None] * num_rows
            continue

        # a) DATE mit 1 Dependency: Ziel > Basis (mind. nächster Tag)
        if ftype == "date" and len(dep_fields) == 1:
            try:
                base_series = pd.to_datetime(columns[dep_fields[0]], errors="coerce")
                start_cfg = pd.to_datetime(cfg.parameterA, errors="coerce") if cfg.parameterA else pd.NaT
                end_cfg   = pd.to_datetime(cfg.parameterB, errors="coerce") if cfg.parameterB else pd.NaT

                out: list[str | None] = []
                for i in range(num_rows):
                    base = base_series.iloc[i]
                    if pd.isna(base):
                        out.append(None)
                        continue

                    start_ord = base.to_pydatetime().date().toordinal() + 1  # > base
                    if not pd.isna(start_cfg):
                        start_ord = max(start_ord, start_cfg.to_pydatetime().date().toordinal())

                    if not pd.isna(end_cfg):
                        end_ord = end_cfg.to_pydatetime().date().toordinal()
                    else:
                        end_ord = start_ord + 30

                    if end_ord < start_ord:
                        end_ord = start_ord

                    o = rng.integers(start_ord, end_ord + 1)
                    out.append(pd.Timestamp.fromordinal(int(o)).date().isoformat())

                columns[field.name] = out
                continue
            except Exception:
                columns[field.name] = [None] * num_rows
                continue

        # b) NUMERIC mit 2 DATE-Dependencies: DwellTime in ganzen Tagen (→ 12, nicht 120)
        if ftype in ("float", "double", "number", "numeric") and len(dep_fields) == 2:
            try:
                # exakte Tage (Zeitanteil abgeschnitten)
                t1 = pd.to_datetime(columns[dep_fields[0]], errors="coerce").to_numpy().astype("datetime64[D]")
                t2 = pd.to_datetime(columns[dep_fields[1]], errors="coerce").to_numpy().astype("datetime64[D]")
                days_int = (t2 - t1).astype("timedelta64[D]").astype(int)

                # Falls mind. 1 Tag gewünscht: days_int = np.maximum(days_int, 1)
                columns[field.name] = [None if not np.isfinite(v) else int(v) for v in days_int]
                continue
            except Exception:
                columns[field.name] = [None] * num_rows
                continue

        # c) STRING/Text mit 1..n Dependencies
        if ftype in ("string", "text"):
            try:
                if dist == "copy":
                    columns[field.name] = list(columns[dep_fields[0]])
                elif dist == "lookup":
                    keys = _split_list(cfg.parameterA or "")
                    vals = _split_list(cfg.parameterB or "")
                    mapping = {k: (vals[i] if i < len(vals) else (vals[-1] if vals else k)) for i, k in enumerate(keys)}
                    default_val = vals[-1] if vals else None
                    base = columns[dep_fields[0]]
                    out = [mapping.get(str(base[i]), default_val if default_val is not None else str(base[i]))
                           for i in range(num_rows)]
                    columns[field.name] = out
                elif dist == "concat":
                    sep = cfg.parameterA or "-"
                    out = [sep.join(str(columns[d][i]) for d in dep_fields) for i in range(num_rows)]
                    columns[field.name] = out
                else:
                    # 🔁 NEU: standardmäßig KEIN simples Kopieren,
                    # sondern deterministische Abbildung abhängig von den Basiswerten.
                    # Optional: eigene Liste in parameterA hinterlegen; sonst Default-Vokabular.
                    vocab = _split_list(cfg.parameterA or "") or ["in", "out", "left", "right", "up", "down"]
                    m = len(vocab)
                    out = []
                    for i in range(num_rows):
                        key = "|".join(str(columns[d][i]) for d in dep_fields)
                        idx = (hash(key) % m)
                        out.append(vocab[idx])
                    columns[field.name] = out
                continue
            except Exception:
                columns[field.name] = [None] * num_rows
                continue

        # d) Fallback
        if len(dep_fields) == 1 and dep_fields[0] in columns:
            columns[field.name] = list(columns[dep_fields[0]])
        else:
            columns[field.name] = [None] * num_rows

    return pd.DataFrame(columns)
