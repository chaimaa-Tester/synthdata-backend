# === Autor: Jan Krämer ===

from typing import List, Optional
from mimesis import *
from mimesis.enums import *
import random
import pandas as pd
import numpy as np
from field_schemas import FrontendField, DistributionConfig

# Vordefinierte Listen an möglichen Werten für Transaktionstypen
# Kredikarten und Währungen, welche vom User noch bearbeitet werden können
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

CREDITCARD_TYPES: List[str] = [
    "VISA Karte",
    "Mastercard",
    "American Express",
    "Girocard (EC)",
    "Maestro"
]

CURRENCY: List [str] = ["EUR", "USD", "CHF", "GBP"]
                        
# Ein Generator zum Erstellen personenbezogener Daten
person_gen = Person(Locale.DE)
# Generator zum Erstellen verschiedener Verteilungen
rng = np.random.default_rng()

def generate_financeData(fields: List[FrontendField], num_rows: int, as_text_for_sheets: bool = False) -> pd.DataFrame:
    """
    Diese Methode enthält die Hauptfunktionalitäten zum Erstellen von Finanzdaten, mit dem SynthData Wizard.
    Es werden die Felder aus dem Frontend ausgelesen und zur Erstellung von einem Finanz-Datensatz verwendet.
    Dabei wird auch überprüft ob eine Verteilung vorliegt, und falls ja mit welchen Parametern diese berechnet
    werden soll. Liegt keine Verteilung vor werden einfach die vorher erstellten Generatoren zur zufälligen
    Generierung von Daten verwendet.
    Es wird außerdem auf Abhängigkeiten geachtet und diese werden entsprechend verarbeitet.

    :param fields: Eine Liste der Felder, welche die Eingaben des Nutzers enthalten.
    :type fields: List[FrontendField]
    :param num_rows: Anzahl der zu erstellenden Reihen im Datensatz.
    :type num_rows: int
    :param as_text_for_sheets: Ob ein Parameter als Text ausgegeben werden soll oder nicht. Nützlich bei der Verwendung von Gleitkommazahlen mit "." oder ",".
    :type as_text_for_sheets: bool
    :return: Ein DataFrame Objekt welches die erzeugten Daten enthält.
    :rtype: DataFrame
    """
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
            # Falls angegeben, werden die Parameter der Verteilung ausgelesen
            paramA = float(dist_config.parameterA) if dist_config else None
            paramB = float(dist_config.parameterB) if dist_config else None

            # Die möglichen Verteilungen pro Feldtyp werden, durch das Frontend, schon vorgegeben
            # und dann hier abgefragt und zum Berechnen der Daten verwendet.
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
            
            # Fallback, falls keine Verteilung angegeben wurde.
            else:
                if field.valueSource == "custom":
                    data = [random.choice(field.customValues) for _ in range(num_rows)]
                else:
                    data = [round(random.uniform(1, 1000), 2) for _ in range(num_rows)]
        
        elif ftype in ["IBAN"]:
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

        elif ftype.lower().startswith("transactiontype"):
            if field.valueSource == "custom":
                data = [random.choice(field.customValues) for _ in range(num_rows)]
            else:
                data = [random.choice(TRANSACTION_TYPES) for _ in range(num_rows)]

        elif ftype == "creditcard":
            if field.valueSource == "custom":
                data = [random.choice(field.customValues) for _ in range(num_rows)]
            else:
                data = [random.choice(CREDITCARD_TYPES) for _ in range(num_rows)]

        elif ftype == "currency":
            if field.valueSource == "custom":
                data = [random.choice(field.customValues) for _ in range(num_rows)]
            else:
                data = [random.choice(CURRENCY) for _ in range(num_rows)]

        else:
            data = [f"{field.name}_{i}" for i in range(num_rows)]   

        columns[field.name] = data

    # 2. Abhängige Felder generieren
    for field in dependent_fields:
        # Es wird zuerst die Abhängigkeit aus den Daten entnommen und Leerzeichen entfernt.
        dep_raw = (field.dependency or "").strip()

        # Fallback, falls keine Abhängigkeit erkannt werden konnte. Gilt als Absicherung sollte aber
        # nicht vorkommen da diese schon im ersten Teil berechnet werden.
        if not dep_raw:
            columns[field.name] = [f"{field.name}_{i}" for i in range(num_rows)]
            continue

        # Es wird der entsprechende Schlüsselwert aus den Spalten gesucht und als Abhängigkeitswert festgelegt.
        # Falls keiner zugeordnet werden kann wird in allen verfügbaren Spalten danach gesucht oder eine
        # Warning geloggt.
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

        # Für die entsprechende abhängige Spalte werden alle zu vergleichenden Werte ausgelesen.
        dep_values = columns[dep_key]
        ftype = (field.type or "").strip()

        # Abhängigkeit ist festgelegt und beinhaltet realitätsnahe Werte, welche pro Überweisungsart
        # in Frage kommen.
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


def _maybe_as_text(seq, as_text: bool) -> list:
    """
    Eine Hilfsmethode welche Gleitkomma Werte in Strings umwandelt. Kann nützlich sein beim Schreiben der Werte
    in CSV oder Excel Sheets da hier auf richtige Schreibweise von 1.000 oder längeren Zahlen geachtet wird.
    (. statt , sepeartor)
    """
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
    Generiert eine realitätsnahe deutsche Kontonummer. Führende 0 ist möglich, da Wert als String zurückgegeben
    wird.

    :return: Kontonummer
    :rtype: string
    """
    k_num = "".join(str(random.randint(0,9)) for _ in range(10))
    return k_num