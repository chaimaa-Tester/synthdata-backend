from typing import List, Optional
from mimesis import Person, Address
from mimesis.enums import Locale, Gender
import random
import pandas as pd
import numpy as np
import re
from field_schemas import FrontendField, DistributionConfig

# Mapping von Frontend-Typen (Aliases) auf Backend-Typen
TYPE_ALIASES = {
    # Basis
    "name": "name", "vorname": "vorname", "nachname": "nachname",
    "geschlecht": "geschlecht", "gender": "geschlecht",
    "alter": "alter", "email": "email", "telefon": "telefon",
    "date": "date",

    # Integer / Kontonummer
    "integer": "integer", "kontonummer": "integer",

    # Zahlen / Float / Betrag / Gewicht / Größe
    "float": "float", "betrag": "float",
    "gewicht": "float", "körpergröße": "float", "body-mass-index": "bmi", "gewichtdiagnose": "gewichtdiagnose",

    # Adressen
    "straße": "straße", "stadt": "stadt", "land": "land", "plz": "plz", "hausnummer": "hausnummer",

    # Transaktionen (Finanzen)
    "transaktionsdatum": "date", "transaktionsart": "string",

    # Containerlogistik (Frontend keys lowercased)
    "unitname": "string",
    "timein": "date",
    "timeout": "date", 
    "Dwelltime in hours": "float",
    "attributesize": "string",
    "attributestatus": "string",
    "attributeweights": "float", 
    "attributedirections": "string",
    "inboundcarrierid": "string",
    "outboundcarrierid": "string",
    "serviceid": "string", 
    "linerid": "string",
}

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


def generate_dummy_data(fields: List[FrontendField], num_rows: int, as_text_for_sheets: bool = False) -> pd.DataFrame:
    columns: dict[str, list] = {}

    # Felder trennen: erst unabhängig, dann abhängig
    independent_fields = [f for f in fields if not (f.dependency or "").strip()]
    dependent_fields = [f for f in fields if (f.dependency or "").strip()]

    # 1) Unabhängige Felder generieren
    for field in independent_fields:
        raw_type = (field.type or "").strip()
        ftype = TYPE_ALIASES.get(raw_type.lower(), raw_type.lower())
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
                # Deterministische exakte Häufigkeitsverteilung, damit kleine Stichproben die Proportionen widerspiegeln
                data = _categorical_exact(values, weights, num_rows)
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
                values = _split_list(paramA) or ["M", "W"]
                weights = [_to_float(w, 1.0) for w in _split_list(paramB or "")]
                if not weights or len(weights) != len(values):
                    weights = [1.0] * len(values)
                weights = np.array(weights, dtype=float)
                weights = weights / weights.sum()
                # Deterministische exakte Häufigkeitsverteilung, damit kleine Stichproben die Proportionen widerspiegeln
                data = _categorical_exact(values, weights, num_rows)
            else:
                
                data = [random.choice(["M", "W"]) for _ in range(num_rows)]

        elif ftype in ["körpergröße", "float", "gewicht"]:
            # Realistische Generierung für Körpergröße (cm) und Gewicht (kg).
            # Modi:
            # - dist == "uniform": Werte uniform zwischen parameterA und parameterB
            # - dist == "normal": parameterA = mean, parameterB = sd (cm)
            # - default: Generiere Körpergröße ~ N(175,10) cm und BMI ~ N(24,4) -> Gewicht = BMI * h^2
            paramA = _to_float(dist_config.parameterA) if dist_config else None
            paramB = _to_float(dist_config.parameterB) if dist_config else None

            if dist == "uniform":
                low = paramA if paramA is not None else 0.0
                high = paramB if paramB is not None else 100.0
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
                arr = np.round(arr.astype(float), 1)  # 1 Dezimalstelle für cm
                data = _maybe_as_text(arr.tolist(), as_text_for_sheets)

            else:
                # Default: Höhen in cm und berechnetes Gewicht aus BMI
                mean_h = 175.0
                sd_h = 10.0
                heights_cm = rng.normal(loc=mean_h, scale=sd_h, size=num_rows)
                heights_cm = np.clip(heights_cm, 140.0, 210.0)
                heights_cm = np.round(heights_cm.astype(float), 1)

                mean_bmi = 24.0
                sd_bmi = 4.0
                bmis = rng.normal(loc=mean_bmi, scale=sd_bmi, size=num_rows)
                bmis = np.clip(bmis, 15.0, 45.0)
                bmis = np.round(bmis.astype(float), 1)

                heights_m = heights_cm / 100.0
                weights = bmis * (heights_m ** 2)
                weights = np.round(weights.astype(float), 1)

                if ftype == "gewicht":
                    data = _maybe_as_text(weights.tolist(), as_text_for_sheets)
                elif ftype == "körpergröße":
                    data = _maybe_as_text(heights_cm.tolist(), as_text_for_sheets)
                else:
                    # Generic float -> gib Gewicht zurück (sinnvoller Default)
                    data = _maybe_as_text(weights.tolist(), as_text_for_sheets)

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
    
    # 2) Abhängige Felder verarbeiten (Dependency nutzen) — jetzt mit Unterstützung für mehrere Dependencies
    for field in dependent_fields:
        dep_raw = (field.dependency or "").strip()

        if not dep_raw:
            columns[field.name] = [f"{field.name}_{i}" for i in range(num_rows)]
            continue

        # parse comma-separated dependency candidates
        candidates = [d.strip() for d in dep_raw.split(",") if d.strip()]
        matched_keys: list[str] = []
        for cand in candidates:
            # exact first
            if cand in columns:
                matched_keys.append(cand)
                continue
            # case-insensitive match
            found = None
            for k in columns.keys():
                if k.lower() == cand.lower():
                    found = k
                    break
            if found:
                matched_keys.append(found)

        # if nothing matched, try a relaxed substring match across column names
        if not matched_keys:
            for cand in candidates:
                for k in columns.keys():
                    if cand.lower() in k.lower() or k.lower() in cand.lower():
                        matched_keys.append(k)
                if matched_keys:
                    break

        if not matched_keys:
            print(f"WARNUNG: Dependency '{dep_raw}' für Feld '{field.name}' nicht gefunden. Verfügbare Spalten: {list(columns.keys())}")
            columns[field.name] = [None] * num_rows
            continue

        # collect dep values map for convenience
        dep_values_map = {k: columns[k] for k in matched_keys}
        raw_type = (field.type or "").strip()
        ftype = TYPE_ALIASES.get(raw_type.lower(), raw_type.lower())

        # Special: DwellTime (abhängig von timeIn & timeOut)
        if "dwell" in field.name.lower() or ftype in ["dwell", "dwelltime", "float"] and "dwell" in raw_type.lower():
            # find time-in/out keys
            time_in_key = None
            time_out_key = None
            for k in matched_keys:
                kl = k.lower()
                if "timein" in kl or ("time" in kl and "in" in kl) or "arrival" in kl:
                    time_in_key = k
                if "timeout" in kl or ("time" in kl and "out" in kl) or "departure" in kl:
                    time_out_key = k
            # fallback: if exactly two matched keys, take first as in and second as out
            if not time_in_key and not time_out_key and len(matched_keys) >= 2:
                time_in_key = matched_keys[0]
                time_out_key = matched_keys[1]

            data = []
            if time_in_key and time_out_key and time_in_key in columns and time_out_key in columns:
                vals_in = columns[time_in_key]
                vals_out = columns[time_out_key]
                for vi, vo in zip(vals_in, vals_out):
                    try:
                        t_in = pd.to_datetime(vi)
                        t_out = pd.to_datetime(vo)
                        if pd.isna(t_in) or pd.isna(t_out) or t_out < t_in:
                            data.append(None)
                        else:
                            hours = (t_out - t_in).total_seconds() / 3600.0
                            data.append(round(float(hours), 2))
                    except Exception:
                        data.append(None)
            else:
                data = [None] * num_rows

            columns[field.name] = _maybe_as_text(data, as_text_for_sheets)
            continue

        # Special: name/vorname/nachname can depend on a gender column among multiple dependencies
        if ftype in ["name", "vorname", "nachname"]:
            # find gender-like dependency among matched_keys
            gender_key = None
            for k in matched_keys:
                if any(x in k.lower() for x in ("geschlecht", "gender", "sex")):
                    gender_key = k
                    break
            # fallback: if first matched looks like gender tokens
            if not gender_key and matched_keys:
                # try to detect by values
                for k in matched_keys:
                    sample = (columns[k][0] if columns[k] else "").strip().lower()
                    if sample in {"m","w","male","female","männlich","weiblich"}:
                        gender_key = k
                        break

            data = []
            if gender_key:
                for gv in columns[gender_key]:
                    gv_norm = (gv or "").strip().lower()
                    male_terms = {"m", "male", "männlich", "mann", "herr"}
                    female_terms = {"w", "female", "weiblich", "frau", "dame"}
                    if gv_norm in male_terms:
                        if ftype == "vorname":
                            data.append(person_gen.first_name(gender=Gender.MALE))
                        elif ftype == "nachname":
                            data.append(person_gen.last_name())
                        else:
                            data.append(f"{person_gen.first_name(gender=Gender.MALE)} {person_gen.last_name()}")
                    elif gv_norm in female_terms:
                        if ftype == "vorname":
                            data.append(person_gen.first_name(gender=Gender.FEMALE))
                        elif ftype == "nachname":
                            data.append(person_gen.last_name())
                        else:
                            data.append(f"{person_gen.first_name(gender=Gender.FEMALE)} {person_gen.last_name()}")
                    else:
                        # fallback: None to highlight mismatch
                        data.append(None)
            else:
                # no gender source: fallback to random gender
                for _ in range(num_rows):
                    if ftype == "vorname":
                        data.append(person_gen.first_name())
                    elif ftype == "nachname":
                        data.append(person_gen.last_name())
                    else:
                        data.append(person_gen.full_name())

            columns[field.name] = data
            continue

        # Default: if multiple dependencies, create a concatenation of their values (string)
        if len(matched_keys) > 1:
            out = []
            lists = [columns[k] for k in matched_keys]
            for i in range(num_rows):
                parts = []
                for lst in lists:
                    v = lst[i] if i < len(lst) else None
                    if v is None:
                        continue
                    parts.append(str(v))
                out.append(" ".join(parts) if parts else None)
            columns[field.name] = out
        else:
            # single dependency: copy values
            columns[field.name] = list(columns[matched_keys[0]])

    return pd.DataFrame(columns)