# generator_general.py
# Autor: CHAIMAA KARIOUI

"""
Projekt: SynthData Wizard
Datei: generator_general.py
Autor: CHAIMAA KARIOUI

Beschreibung:
Dieses Modul erzeugt synthetische "Allgemeine Daten" (General Use Case) für den SynthData Wizard.
Die Daten werden anhand der vom Frontend übergebenen Felddefinitionen (FrontendField) generiert.
Die Generierung nutzt mimesis (Person/Address) sowie faker für verschiedene Datentypen.

Klassendokumentation (Modul-/Komponentenübersicht):
- Keine Klassen im Modul, nur Funktionen und Generator-Instanzen.
- Globale Generatoren:
  - person_gen: mimesis Person-Generator (Locale.DE)
  - address_gen: mimesis Address-Generator (Locale.DE)
  - faker: Faker-Instanz (zusätzliche synthetische Werte)
  - rng: numpy Zufallsgenerator (aktuell in dieser Funktion nicht genutzt)
- Hauptfunktion:
  - generate_generalData(fields, num_rows, as_text_for_sheets=False) -> pd.DataFrame

Implementierungsdokumentation (Designentscheidungen):
- Felder werden in unabhängige und abhängige Felder getrennt.
  - Unabhängige Felder: dependency ist leer -> werden direkt erzeugt.
  - Abhängige Felder: dependency ist gesetzt -> sind für spätere Implementierung vorgesehen.
- In dieser Version werden nur unabhängige Felder erzeugt; abhängige Felder werden noch nicht berechnet.
- Pro Feldtyp wird eine passende Generator-Funktion aufgerufen (mimesis/faker/random).
- Die Ausgabe ist ein DataFrame, dessen Spaltennamen aus field.name stammen.
"""

from typing import List
from mimesis import Person, Address
from mimesis.enums import Locale
from faker import Faker
import random
import pandas as pd
import numpy as np
from field_schemas import FrontendField

# Globale Generator-Instanzen (Locale.DE für deutschsprachige Daten)
person_gen = Person(Locale.DE)
address_gen = Address(Locale.DE)
faker = Faker()

# Globaler RNG (für mögliche spätere Erweiterungen / deterministische Seeds)
rng = np.random.default_rng()


def generate_generalData(
    fields: List[FrontendField],
    num_rows: int,
    as_text_for_sheets: bool = False
) -> pd.DataFrame:
    """
    Zweck:
    Generiert synthetische Daten für den Use Case "Allgemeine Daten" basierend
    auf einer Liste von FrontendField-Felddefinitionen.

    Args:
        fields:
            Liste der gewünschten Felder aus dem Frontend.
            Relevante Attribute eines FrontendField:
            - name: Spaltenname im Output
            - type: Feldtyp (z. B. "string", "firstname", "postcode", ...)
            - dependency: optional, beschreibt Abhängigkeiten zu anderen Feldern
            - distributionConfig: optional, Verteilungsparameter (in dieser Version nicht ausgewertet)
        num_rows:
            Anzahl der zu erzeugenden Zeilen.
        as_text_for_sheets:
            Optionales Flag für XLSX/Sheet-Export (in dieser Version ohne Wirkung).
            Für spätere Erweiterung gedacht, um Datentypen als Text zu formatieren.

    Returns:
        pd.DataFrame:
            DataFrame mit num_rows Zeilen und einer Spalte pro Felddefinition.

    Implementierungsdokumentation (Ablauf):
    1) Felder werden nach Abhängigkeiten getrennt:
       - independent_fields: dependency ist leer -> sofort generierbar
       - dependent_fields: dependency ist gesetzt -> spätere Implementierung (aktuell nicht verwendet)

    2) Für jedes unabhängige Feld wird anhand von field.type eine Datenliste erzeugt.
       - Primäre Datengeneratoren:
         - mimesis (person_gen, address_gen) für personenbezogene/adressbezogene Werte
         - faker für diverse synthetische Daten
         - random für einfache Zufallswahl
    3) Ergebnis wird als dict[str, list] gesammelt und als DataFrame zurückgegeben.
    """
    columns: dict[str, list] = {}

    # Felder trennen: erst unabhängig, dann abhängig (abhängig ist vorbereitet, aber noch nicht implementiert)
    independent_fields = [f for f in fields if not (f.dependency or "").strip()]
    dependent_fields = [f for f in fields if (f.dependency or "").strip()]

    # Hinweis: Abhängige Felder werden in dieser Version noch nicht verarbeitet.
    _ = dependent_fields

    # 1) Unabhängige Felder generieren
    for field in independent_fields:
        ftype = (field.type or "").strip().lower()
        dist_config = getattr(field, "distributionConfig", None)
        dist = (getattr(dist_config, "distribution", "") or "").strip().lower() if dist_config else ""

        # Primitive Datentypen
        if ftype == "string":
            data = [faker.word() for _ in range(num_rows)]

        elif ftype == "number":
            data = [faker.random_number() for _ in range(num_rows)]

        elif ftype == "boolean":
            # Korrektur: random.choice erwartet eine Sequenz (Liste/Tuple), nicht zwei Argumente.
            data = [random.choice([True, False]) for _ in range(num_rows)]

        elif ftype == "date":
            # Faker liefert Strings im angegebenen Pattern
            data = [faker.date(pattern="%d.%m.%Y") for _ in range(num_rows)]

        elif ftype == "time":
            # Faker nutzt strftime-Pattern
            data = [faker.time(pattern="%H:%M:%S") for _ in range(num_rows)]

        elif ftype == "datetime":
            data = [faker.date_time() for _ in range(num_rows)]

        # Personenbezogene Daten
        elif ftype == "firstname":
            data = [person_gen.first_name() for _ in range(num_rows)]

        elif ftype == "lastname":
            data = [person_gen.last_name() for _ in range(num_rows)]

        elif ftype == "fullname":
            data = [person_gen.full_name() for _ in range(num_rows)]

        elif ftype == "gender":
            data = [person_gen.gender() for _ in range(num_rows)]

        # Kommunikationsdaten
        elif ftype == "email":
            # mimesis.email akzeptiert optional eine domain-Liste
            data = [
                person_gen.email(["web.de", "outlook.com", "yahoo.com", "gmail.com", "icloud.com"])
                for _ in range(num_rows)
            ]

        elif ftype in ("phone", "telefon"):
            data = [person_gen.phone_number() for _ in range(num_rows)]

        # Adressdaten
        elif ftype == "street":
            data = [address_gen.street_name() for _ in range(num_rows)]

        elif ftype == "house_number":
            data = [address_gen.street_number() for _ in range(num_rows)]

        elif ftype == "postcode":
            data = [address_gen.postal_code() for _ in range(num_rows)]

        elif ftype == "city":
            data = [address_gen.city() for _ in range(num_rows)]

        elif ftype == "state":
            data = [address_gen.state() for _ in range(num_rows)]

        elif ftype == "country":
            data = [address_gen.country() for _ in range(num_rows)]

        elif ftype == "full_address":
            data = [address_gen.address() for _ in range(num_rows)]

        # Kategorien & Listen
        elif ftype == "enum":
            """
            Hinweis:
            faker hat keine generische faker.enum()-Methode im Standardumfang.
            Diese Stelle ist als Platzhalter zu verstehen.
            Falls ein echtes Enum aus dem Frontend kommen soll, sollte hier
            field.customValues / valueSource ausgewertet werden.
            """
            data = [f"enum_{i}" for i in range(num_rows)]

        elif ftype == "list":
            # Beispiel: pro Zeile eine Liste von Werten
            data = [[faker.word(), faker.random_number()] for _ in range(num_rows)]

        # Identifikatoren
        elif ftype == "uuid":
            data = [faker.uuid4() for _ in range(num_rows)]

        # Gesundheitsdaten
        elif ftype == "body_height":
            data = [person_gen.height() for _ in range(num_rows)]

        elif ftype == "weight":
            data = [person_gen.weight() for _ in range(num_rows)]

        else:
            # Fallback: generische Werte, falls field.type unbekannt ist
            data = [f"{field.name}_{i}" for i in range(num_rows)]

        columns[field.name] = data

    return pd.DataFrame(columns)