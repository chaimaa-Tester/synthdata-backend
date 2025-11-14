import pandas as pd
import random
from faker import Faker

# Faker Deutsch
fake = Faker("de_DE")

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
    return "gesund" if gesund >= 2 else "krank"

def generate_healthData(rows, rowCount):
    data = []
    versicherungsarten = ["gesetzlich", "privat"]

    for _ in range(rowCount):

        # --- Unbegrenzte deutsche Namen ---
        geschlecht = random.choice(["m", "w"])

        if geschlecht == "m":
            name = fake.first_name_male() + " " + fake.last_name()
        else:
            name = fake.first_name_female() + " " + fake.last_name()

        # Versicherung
        versicherung = random.choices(["gesetzlich", "privat"], weights=[0.75, 0.25])[0]

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
        allergien = random.choice(["keine", "Pollen", "Penicillin", "Hausstaub", "Nüsse", "Krebs"])

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