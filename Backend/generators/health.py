# generator_health.py
# Autor: MAROUANE BOUHAMZA
"""
Projekt: SynthData Wizard
Datei: generator_health.py
Autor: Marouane Bouhamza

Beschreibung:
Dieses Modul erzeugt einen künstlichen Datensatz (synthetic dataset) für
Gesundheits- bzw. Körperdaten. Die Daten werden zufällig generiert und
anschließend – falls angefordert – um abgeleitete Werte wie den BMI und
den BMI-Status ergänzt.

Ziel des Moduls:
- Simulation realistischer Gesundheitsdaten
- Flexible Generierung basierend auf Frontend-Anforderungen
- Demonstration von Datenabhängigkeiten (z. B. BMI basiert auf Größe/Gewicht)

Verwendete Technologien:
- Python
- Pandas (Tabellenstruktur)
- Faker (synthetische Daten)
- Random (Zufallswerte)

Datenlogik:
1. Unabhängige Daten werden zuerst erzeugt
   (z.B. Körpergröße, Gewicht)

2. Abhängige Daten werden anschließend berechnet
   (z.B. BMI, BMI-Kategorie)

Die Funktion liefert am Ende einen pandas DataFrame.
"""

import pandas as pd
import random
from faker import Faker
from typing import List
from field_schemas import FrontendField

# Initialisierung der Faker-Bibliothek mit deutscher Lokalisierung
fake = Faker("de_DE")


# ---------------------------------------------------------
# BMI Berechnung
# ---------------------------------------------------------
def berechne_bmi(gewicht, groesse):
    """
    Berechnet den Body-Mass-Index.

    Formel:
    BMI = Gewicht (kg) / Größe² (m)

    Parameter
    ----------
    gewicht : float
        Körpergewicht in Kilogramm
    groesse : float
        Körpergröße in Metern

    Rückgabe
    --------
    float
        BMI-Wert gerundet auf eine Nachkommastelle
    """
    return round(gewicht / (groesse ** 2), 1)


# ---------------------------------------------------------
# BMI Klassifikation
# ---------------------------------------------------------
def kategorisiere_bmi(bmi):
    """
    Ordnet einen BMI-Wert einer WHO-Kategorie zu.

    Kategorien laut Weltgesundheitsorganisation:

    Untergewicht      < 18.5
    Normalgewicht     18.5 – 24.9
    Übergewicht       25 – 29.9
    Adipositas Grad I 30 – 34.9
    Adipositas Grad II 35 – 39.9
    Adipositas Grad III >= 40

    Parameter
    ----------
    bmi : float

    Rückgabe
    --------
    str
        Kategorie des BMI
    """

    if bmi < 18.5:
        return "Untergewicht"

    elif 18.5 <= bmi < 25.0:
        return "Normalgewicht"

    elif 25.0 <= bmi < 30.0:
        return "Übergewicht"

    elif 30.0 <= bmi < 35.0:
        return "Adipositas (Grad I)"

    elif 35.0 <= bmi < 40.0:
        return "Adipositas (Grad II)"

    else:
        return "Adipositas (Grad III)"


# ---------------------------------------------------------
# Hauptfunktion zur Datengenerierung
# ---------------------------------------------------------
def generate_healthData(rows: List[FrontendField], rowCount: int):
    """
    Erstellt einen synthetischen Datensatz basierend auf
    den angeforderten Feldern aus dem Frontend.

    Funktionsprinzip
    ----------------
    1. Trennung der Felder in:
       - unabhängige Felder
       - abhängige Felder

    2. Zuerst werden Basisdaten erzeugt.

    3. Danach werden berechnete Werte erzeugt.

    Parameter
    ----------
    rows : List[FrontendField]
        Liste der gewünschten Datenfelder.

    rowCount : int
        Anzahl der zu generierenden Datensätze.

    Rückgabe
    --------
    pandas.DataFrame
        Tabelle mit den generierten Daten.
    """

    # Wenn keine Felder übergeben wurden, wird ein leerer DataFrame zurückgegeben
    if not rows:
        return pd.DataFrame()

    # Dictionary zum Sammeln der generierten Spalten
    columns = {}

    # Aufteilen in unabhängige und abhängige Felder
    independent_fields = [
        f for f in rows if f.type.lower() not in ["bmi", "bmi-status"]
    ]

    dependent_fields = [
        f for f in rows if f.type.lower() in ["bmi", "bmi-status"]
    ]

    # ---------------------------------------------------------
    # Schritt 1: Generierung der unabhängigen Daten
    # ---------------------------------------------------------
    for field in independent_fields:

        ftype = field.type.lower()

        # Generierung zufälliger Körpergrößen
        if ftype == "body_height":

            # Werte zwischen 1.50m und 2.00m
            data = [
                round(random.uniform(1.50, 2.00), 2)
                for _ in range(rowCount)
            ]

        # Generierung zufälliger Gewichte
        elif ftype == "weight":

            # Werte zwischen 45kg und 120kg
            data = [
                random.randint(45, 120)
                for _ in range(rowCount)
            ]

        # Fallback falls ein unbekannter Feldtyp angefragt wurde
        else:

            data = [
                f"{field.name}_{i}"
                for i in range(rowCount)
            ]

        # Speicherung der Spalte
        columns[field.name] = data

    # ---------------------------------------------------------
    # Schritt 2: Generierung der abhängigen Daten
    # ---------------------------------------------------------
    for field in dependent_fields:

        ftype = field.type.lower()

        # -----------------------------------------------------
        # BMI Berechnung
        # -----------------------------------------------------
        if ftype == "bmi":

            # Suche nach vorhandenen Spalten für Größe und Gewicht
            groesse_col = next(
                (f.name for f in independent_fields if f.type.lower() == "body_height"),
                None
            )

            gewicht_col = next(
                (f.name for f in independent_fields if f.type.lower() == "weight"),
                None
            )

            # BMI kann nur berechnet werden wenn beide Werte existieren
            if groesse_col and gewicht_col:

                bmi_data = []

                for i in range(rowCount):

                    groesse = columns[groesse_col][i]
                    gewicht = columns[gewicht_col][i]

                    bmi = berechne_bmi(gewicht, groesse)

                    bmi_data.append(bmi)

                columns[field.name] = bmi_data

            else:
                # Falls Daten fehlen -> None
                columns[field.name] = [None] * rowCount

        # -----------------------------------------------------
        # BMI Status
        # -----------------------------------------------------
        elif ftype == "bmi-status":

            bmi_col = next(
                (f.name for f in rows if f.type.lower() == "bmi"),
                None
            )

            if bmi_col and bmi_col in columns:

                status_data = [
                    kategorisiere_bmi(bmi)
                    for bmi in columns[bmi_col]
                ]

                columns[field.name] = status_data

            else:
                columns[field.name] = [None] * rowCount

    # Erstellung des finalen DataFrames
    return pd.DataFrame(columns)

    return pd.DataFrame(columns)
