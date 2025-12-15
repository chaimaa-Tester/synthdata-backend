import pandas as pd
import random
from faker import Faker
from typing import List
from field_schemas import FrontendField

# Faker Deutsch
fake = Faker("de_DE")

def berechne_bmi(gewicht, groesse):
    """Berechnet den Body-Mass-Index (BMI) in kg/m^2."""
    # Formel: BMI = Gewicht [kg] / (Größe [m])^2
    return round(gewicht / (groesse ** 2), 1)

def kategorisiere_bmi(bmi):
    """
    Kategorisiert den BMI-Wert gemäß den WHO-Standards.
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
    else: # bmi >= 40.0
        return "Adipositas (Grad III)"

def generate_healthData(rows: List[FrontendField], rowCount: int):
    """
    Generiert einen synthetischen DataFrame basierend auf den angeforderten Feldern.
    BMI und BMI-Status werden nur generiert, wenn sie explizit angefragt sind.
    """
    if not rows:
        return pd.DataFrame()
    
    columns = {}
    
    # Felder trennen: unabhängig vs. abhängig
    independent_fields = [f for f in rows if f.type.lower() not in ["bmi", "bmi-status"]]
    dependent_fields = [f for f in rows if f.type.lower() in ["bmi", "bmi-status"]]
    
    # 1. Unabhängige Felder generieren
    for field in independent_fields:
        ftype = field.type.lower()
        
        if ftype == "body_height":
            data = [round(random.uniform(1.50, 2.00), 2) for _ in range(rowCount)]
        elif ftype == "weight":
            data = [random.randint(45, 120) for _ in range(rowCount)]
        else:
            data = [f"{field.name}_{i}" for i in range(rowCount)]  # Fallback
        
        columns[field.name] = data
    
    # 2. Abhängige Felder generieren (BMI und BMI-Status)
    for field in dependent_fields:
        ftype = field.type.lower()
        
        if ftype == "bmi":
            # BMI berechnen, wenn Größe und Gewicht vorhanden
            groesse_col = next((f.name for f in independent_fields if f.type.lower() == "größe"), None)
            gewicht_col = next((f.name for f in independent_fields if f.type.lower() == "gewicht"), None)
            if groesse_col and gewicht_col:
                bmi_data = []
                for i in range(rowCount):
                    groesse = columns[groesse_col][i]
                    gewicht = columns[gewicht_col][i]
                    bmi = berechne_bmi(gewicht, groesse)
                    bmi_data.append(bmi)
                columns[field.name] = bmi_data
            else:
                columns[field.name] = [None] * rowCount  # Oder Fallback
        
        elif ftype == "bmi-status":
            # BMI-Status berechnen, wenn BMI vorhanden
            bmi_col = next((f.name for f in rows if f.type.lower() == "bmi"), None)
            if bmi_col and bmi_col in columns:
                status_data = [kategorisiere_bmi(bmi) for bmi in columns[bmi_col]]
                columns[field.name] = status_data
            else:
                columns[field.name] = [None] * rowCount  # Oder Fallback
    
    return pd.DataFrame(columns)