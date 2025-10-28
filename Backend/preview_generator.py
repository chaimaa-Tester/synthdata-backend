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
    "timein": "date", "timeout": "date",
    "attributesizes": "string", "attributesize": "string",
    "attributes": "string", "attributestatus": "string", "attributestatuses": "string",
    "attributeweights": "float", "attributedirections": "string",
    "inboundcarrierid": "string", "outboundcarrierid": "string",
    "serviceid": "string", "linerid": "string",
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
    
    # 2) Abhängige Felder verarbeiten (Dependency nutzen)
    for field in dependent_fields:
        dep_raw = (field.dependency or "").strip()

        if not dep_raw:
            columns[field.name] = [f"{field.name}_{i}" for i in range(num_rows)]
            continue

        # Dependency-Key case-insensitiv auflösen; comma-separierte Einträge unterstützen und erstes Matching verwenden
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
        raw_type = (field.type or "").strip()
        ftype = TYPE_ALIASES.get(raw_type.lower(), raw_type.lower())

        if ftype in ["name", "vorname", "nachname"]:
            data = []
            male_terms = {"m", "männlich", "mann", "herr"}
            female_terms = {"w" , "weiblich", "frau", "dame"}
            for gv in dep_values:
                gv_norm = (gv or "").strip().lower()

                if gv_norm in male_terms:
                    # männlich
                    if ftype == "vorname":
                        data.append(person_gen.first_name(gender=Gender.MALE))
                    elif ftype == "nachname":
                        data.append(person_gen.last_name())
                    else:
                        first = person_gen.first_name(gender=Gender.MALE)
                        last = person_gen.last_name()
                        data.append(f"{first} {last}")

                elif gv_norm in female_terms:
                    # weiblich
                    if ftype == "vorname":
                        data.append(person_gen.first_name(gender=Gender.FEMALE))
                    elif ftype == "nachname":
                        data.append(person_gen.last_name())
                    else:
                        first = person_gen.first_name(gender=Gender.FEMALE)
                        last = person_gen.last_name()
                        data.append(f"{first} {last}")

                else:
                    # Wenn der Dep-Wert wie ein vollständiger Name aussieht (z.B. 'Karl Gotti'),
                    # dann wurde vermutlich das Namensfeld statt des Geschlechtsfeldes als Dependency angegeben.
                    gv_str = (gv or "").strip()
                    name_like = False
                    if gv_str:
                        #ich findeINTERESSANT: Wir verwenden hier eine erweiterte Heuristik, um vollständige Namen zu erkennen
                        # Erweiterte Heuristik: mehrere Namensbestandteile erlaubt, Bindestriche und
                        # apostrophartige Zeichen werden unterstützt (z.B. "Anne-Marie O'Neill").
                        # Mindestens zwei Wörter, jeweils mit Großbuchstaben beginnend.
                        name_like = bool(re.match(r"^[A-ZÄÖÜ][A-Za-zäöüß'’\-]+(?:\s+[A-ZÄÖÜ][A-Za-zäöüß'’\-]+)+$", gv_str))

                    if name_like:
                        print(f"WARNUNG: Dependency-Spalte '{dep_key}' scheint vollständige Namen zu enthalten ('{gv_str}'). Feld '{field.name}' erwartet ein Geschlecht. Bitte setze die Dependency auf das Geschlechtsfeld.")
                        # Deutlichere Signalwirkung: keine automatische Zuweisung, stattdessen None
                        data.append(None)
                    else:
                        # FALLBACK: Unbekannter Wert -> verwende weiblich als Fallback 
                        print(f"WARNUNG: Unbekanntes Geschlecht '{gv}', verwende weiblich als Fallback")
                        if ftype == "vorname":
                            data.append(person_gen.first_name(gender=Gender.FEMALE))
                        elif ftype == "nachname":
                            data.append(person_gen.last_name())
                        else:
                            first = person_gen.first_name(gender=Gender.FEMALE)
                            last = person_gen.last_name()
                            data.append(f"{first} {last}")

            columns[field.name] = data
        else:
            columns[field.name] = list(dep_values)

    return pd.DataFrame(columns)