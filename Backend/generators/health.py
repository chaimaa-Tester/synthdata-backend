import pandas as pd
import random

def berechne_bmi(gewicht, groesse):
    return round(gewicht / (groesse ** 2), 1)

def berechne_map(sys, dia):
    return round((sys + 2 * dia) / 3, 1)

def berechne_gfr(alter, gewicht, kreatinin, geschlecht):
    gfr = ((140 - alter) * gewicht) / (72 * kreatinin)
    if geschlecht == "w":
        gfr *= 0.85
    return round(gfr, 1)

def bewerte_patient(bmi, map_wert, gfr):
    gesund = 0
    if 18.5 <= bmi <= 25:
        gesund += 1
    if 70 <= map_wert <= 105:
        gesund += 1
    if gfr > 60:
        gesund += 1
    return "gesund" if gesund >= 2 else "kritisch"

def generate_healthData(rows, rowCount):
    # Name-Geschlecht-Zuordnung
    name_gender_map = {
        "Max": "m", "Paul": "m", "Lukas": "m", "Leon": "m", "Tim": "m",
        "Anna": "w", "Mia": "w", "Sophie": "w", "Laura": "w", "Emma": "w"
    }

    # Nur so viele Namen wie rowCount erlaubt
    if rowCount > len(name_gender_map):
        raise ValueError("rowCount überschreitet verfügbare eindeutige Namen")

    # Zufällige Auswahl ohne Wiederholung
    selected_names = random.sample(list(name_gender_map.keys()), rowCount)
    data = []
    versicherungsarten = ["gesetzlich", "privat"]
    versicherung = random.choices(
        versicherungsarten,
        weights=[0.75, 0.25],  # ca. 75% gesetzlich, 25% privat
        k=1
    )[0]
    for name in selected_names:
        geschlecht = name_gender_map[name]
        alter = random.randint(18, 90)
        groesse = round(random.uniform(1.50, 2.00), 2)
        gewicht = random.randint(45, 120)
        rr_sys = random.randint(100, 160)
        rr_dia = random.randint(60, 100)
        puls = random.randint(55, 100)
        temperatur = round(random.uniform(36.0, 38.5), 1)
        spo2 = round(random.uniform(92.0, 100.0), 1)
        kreatinin = round(random.uniform(0.6, 1.5), 2)
        raucher = random.choice(["ja", "nein"])
        diabetes = random.choice(["ja", "nein"])
        allergien = random.choice(["keine", "Pollen", "Penicillin", "Hausstaub", "Nüsse","Krebs"])

        bmi = berechne_bmi(gewicht, groesse)
        map_wert = berechne_map(rr_sys, rr_dia)
        gfr = berechne_gfr(alter, gewicht, kreatinin, geschlecht)
        status = bewerte_patient(bmi, map_wert, gfr)

        data.append({
            "Name": name,
            "Geschlecht": geschlecht,
            "Versicherung": versicherung,
            "Alter": alter,
            "Größe_m": groesse,
            "Gewicht_kg": gewicht,
            "RR_sys_mmHg": rr_sys,
            "RR_dia_mmHg": rr_dia,
            "MAP_mmHg": map_wert,
            "Puls_bpm": puls,
            "Temperatur_C": temperatur,
            "SpO2_%": spo2,
            "Kreatinin_mg/dl": kreatinin,
            "BMI": bmi,
            "GFR": gfr,
            "Raucher": raucher,
            "Diabetes": diabetes,
            "Allergien": allergien,
            "Status": status
        })

    return pd.DataFrame(data)