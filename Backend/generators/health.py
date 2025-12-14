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
    Generiert einen synthetischen DataFrame mit Größe, Gewicht, BMI und BMI-Status.
    BMI wird abhängig von Gewicht und Größe berechnet.
    """
    data = []
    
    # Felder für Gewicht und Größe identifizieren (falls definiert)
    weight_field = next((f for f in rows if f.type.lower() == "weight"), None)
    height_field = next((f for f in rows if f.type.lower() == "body_height"), None)
    bmi_field = next((f for f in rows if f.type.lower() == "bmi"), None)
    bmi_status_field = next((f for f in rows if f.type.lower() == "bmi-status"), None)
    
    for _ in range(rowCount):
        # Gewicht und Größe generieren (zufällig oder basierend auf Feldern)
        if weight_field:
            # Hier könntest du eine Verteilung aus weight_field.distributionConfig verwenden
            gewicht = random.randint(45, 120)  # Placeholder; passe an Verteilung an
        else:
            gewicht = random.randint(45, 120)
        
        if height_field:
            groesse = round(random.uniform(1.50, 2.00), 2)  # Placeholder
        else:
            groesse = round(random.uniform(1.50, 2.00), 2)
        
        # BMI abhängig berechnen
        bmi = berechne_bmi(gewicht, groesse)
        bmi_status = kategorisiere_bmi(bmi)
        
        # Daten sammeln
        row_data = {}
        if height_field:
            row_data[height_field.name] = groesse
        else:
            row_data["Größe_m"] = groesse
        
        if weight_field:
            row_data[weight_field.name] = gewicht
        else:
            row_data["Gewicht_kg"] = gewicht

        if bmi_field:
            row_data[bmi_field.name] = bmi
        else:
            row_data["BMI"] = bmi
        
        if bmi_status_field:
            row_data[bmi_status_field.name] = bmi_status
        else:
            row_data["BMI_Status"] = bmi_status
        
        data.append(row_data)
    
    return pd.DataFrame(data)