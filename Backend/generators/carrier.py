# generator_carrier.py
# Autor: CHAIMAA KARIOUI

"""
Projekt: SynthData Wizard
Datei: generator_carrier.py
Autor: CHAIMAA KARIOUI

Beschreibung:
Dieses Modul erzeugt synthetische Carrier-/Schiffsdaten für den Use Case "Logistik".
Die generierten Daten repräsentieren typische Informationen aus der Containerlogistik,
z. B. Reederei (Liner), Service-Name/Route, Schiffsname sowie ETA/ETD und Kennzahlen.

Enthaltene Datenbausteine:
- LINERS: Mapping von Reederei-IDs auf Reederei-Namen
- SERVICE_DETAILS: Mapping von Service-IDs auf Service-Metadaten (Name, Route, linerId)
- SERVICES_BY_LINER: Ableitung (Gruppierung) von Service-IDs je Reederei
- SHIP_NAMES: Liste realistischer Schiffsnamen zur zufälligen Auswahl

Hauptfunktion:
- generate_carrierData: erzeugt ein pandas.DataFrame mit Carrier-/Schiffsdaten

Implementierungsdokumentation:
- Zufallszahlen werden über numpy.random.default_rng(seed) erzeugt (reproduzierbar per Seed).
- Services werden konsistent zur zufällig gezogenen Reederei gewählt (über SERVICES_BY_LINER).
- ETA/ETD werden als Zeitpaare erzeugt: ETD = ETA + Liegezeit (12–120 Stunden).
- Kennzahlen (TEU und Länge) werden als integer Arrays erzeugt.
"""

from typing import List, Dict
import pandas as pd
import numpy as np

# -------------------------------
# Konfiguration - Reedereien
# -------------------------------

"""
LINERS

Zweck:
Stellt eine zentrale Liste von Reedereien (Liner) bereit.

Implementierung:
- Keys sind integer IDs (stabil, maschinenlesbar).
- Values sind Anzeigenamen (menschenlesbar).
"""
LINERS: Dict[int, str] = {
    1: "Maersk",
    2: "Hapag-Lloyd",
    3: "MSC",
    4: "CMA CGM",
    5: "Evergreen",
    6: "ONE (Ocean Network Express)",
    7: "Yang Ming",
    8: "ZIM",
    9: "COSCO",
    10: "Hamburg Süd",
}

# -------------------------------
# Konfiguration - Services & Routen
# -------------------------------

"""
SERVICE_DETAILS

Zweck:
Definiert Service-Linien (ServiceName) und deren Routen (ServiceRoute) sowie
die zugehörige Reederei (linerId).

Implementierung:
- Key: service_id (int)
- Value: dict mit Feldern:
  - name: Servicebezeichnung
  - route: Route (Start–Ziel)
  - linerId: Reederei-ID aus LINERS
"""
SERVICE_DETAILS: Dict[int, Dict[str, object]] = {
    # Maersk Services
    70: {"name": "AE1 Europe–Asia", "route": "Rotterdam – Shanghai", "linerId": 1},
    71: {"name": "AE5 Europe–China", "route": "Hamburg – Shenzhen", "linerId": 1},
    72: {"name": "Baltic Express", "route": "Gdansk – Bremerhaven", "linerId": 1},

    # Hapag-Lloyd Services
    66: {"name": "Atlantic Shuttle", "route": "Hamburg – New York", "linerId": 2},
    67: {"name": "Asia Express", "route": "Hamburg – Singapore", "linerId": 2},
    68: {"name": "Pacific Bridge", "route": "Rotterdam – Los Angeles", "linerId": 2},

    # MSC Services
    50: {"name": "Mediterranean Loop", "route": "Valencia – Piraeus", "linerId": 3},
    51: {"name": "South China Route", "route": "Hong Kong – Ningbo", "linerId": 3},
    52: {"name": "Transatlantic East", "route": "Bremerhaven – Halifax", "linerId": 3},

    # Evergreen Services
    40: {"name": "Red Sea Express", "route": "Jeddah – Port Said", "linerId": 5},
    41: {"name": "Europe–India Line", "route": "Hamburg – Mumbai", "linerId": 5},

    # ONE Services
    80: {"name": "Far East Connector", "route": "Tokyo – Busan – Shanghai", "linerId": 6},
    81: {"name": "Pacific Connector", "route": "Yokohama – Long Beach", "linerId": 6},

    # ZIM Services
    90: {"name": "Mediterranean Expr.", "route": "Haifa – Genoa", "linerId": 8},
    91: {"name": "North Africa Feeder", "route": "Valencia – Casablanca", "linerId": 8},
}

"""
SERVICES_BY_LINER

Zweck:
Gruppiert Services nach Reederei-ID, um später "passende Services"
zur ausgewählten Reederei ziehen zu können.

Implementierung:
- Wird aus SERVICE_DETAILS automatisch abgeleitet.
- Key: liner_id
- Value: Liste der service_ids, die zu dieser Reederei gehören.
"""
SERVICES_BY_LINER: Dict[int, List[int]] = {}
for service_id, service_data in SERVICE_DETAILS.items():
    liner_id = service_data["linerId"]
    SERVICES_BY_LINER.setdefault(liner_id, []).append(service_id)

# -------------------------------
# Konfiguration - Schiffsnamen
# -------------------------------

"""
SHIP_NAMES

Zweck:
Liste realistischer Schiffsnamen zur zufälligen Auswahl.

Implementierung:
- Wird per rng.choice zufällig gezogen.
- replace=True erlaubt Wiederholungen (realistisch bei großen rowCounts).
"""
SHIP_NAMES = [
    "MSC Gülsün", "MSC Zoe", "Maersk Hamburg", "Maersk Sevilla",
    "CMA CGM Jacques Saadé", "CMA CGM Alexander", "Ever Ace", "Ever Given",
    "Hapag-Lloyd Dalian Express", "Chicago Express", "ONE Infinity", "ONE Triumph",
    "ZIM Ningbo", "COSCO Harmony", "Yang Ming Orchid", "Hamburg Süd Rio Bravo",
    "Neptune Pioneer", "Ocean Serenity", "Atlantic Horizon", "Pacific Voyager",
    "Northern Spirit", "Aurora Blue", "Coral Star", "Golden Horizon",
    "Silver Wave", "Sunrise Carrier", "Baltic Spirit", "Mariner's Hope",
]

# -------------------------------
# Hilfsfunktionen
# -------------------------------

def _get_service_for_liner(liner_id: int, rng: np.random.Generator) -> tuple[str, str]:
    """
    Zweck:
    Wählt für eine gegebene Reederei (liner_id) einen passenden Service (Name + Route).

    Args:
        liner_id: Reederei-ID (Key aus LINERS)
        rng: numpy Zufallsgenerator (reproduzierbar per Seed)

    Returns:
        (service_name, service_route)
        - service_name: Name des Services
        - service_route: Route des Services

    Implementierung:
    - Liest alle Service-IDs aus SERVICES_BY_LINER[liner_id].
    - Falls vorhanden: zufällige Auswahl eines Service, Rückgabe von name/route.
    - Falls nicht vorhanden: Fallback-Werte (Generic Line / Unbekannte Route).
    """
    available_services = SERVICES_BY_LINER.get(liner_id, [])

    if available_services:
        service_id = rng.choice(available_services)
        service_data = SERVICE_DETAILS[service_id]
        return service_data["name"], service_data["route"]
    else:
        return "Generic Line", "Unbekannte Route"


def _generate_ship_times(
    row_count: int,
    rng: np.random.Generator
) -> tuple[List[pd.Timestamp], List[pd.Timestamp]]:
    """
    Zweck:
    Erzeugt realistische ETA/ETD-Zeitpaare für row_count Einträge.

    Args:
        row_count: Anzahl der Zeitpaare
        rng: numpy Zufallsgenerator

    Returns:
        (eta_list, etd_list)
        - eta_list: Liste von pd.Timestamp (Ankunftszeit)
        - etd_list: Liste von pd.Timestamp (Abfahrtszeit)

    Implementierung:
    - base_start: fester Startpunkt als Referenzdatum
    - ETA wird zufällig innerhalb der nächsten 120 Tage (plus zufällige Stunde) erzeugt.
    - ETD wird als ETA + Liegezeit berechnet.
      Die Liegezeit (dwell_hours) liegt zwischen 12 und 120 Stunden.
    """
    base_start = pd.Timestamp("2025-01-01")
    eta_list, etd_list = [], []

    for _ in range(row_count):
        eta = base_start + pd.Timedelta(
            days=rng.integers(0, 120),
            hours=rng.integers(0, 24),
        )

        dwell_hours = rng.integers(12, 120)
        etd = eta + pd.Timedelta(hours=dwell_hours)

        eta_list.append(eta)
        etd_list.append(etd)

    return eta_list, etd_list


def _generate_ship_metrics(
    row_count: int,
    rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Zweck:
    Erzeugt Kennzahlen für Schiffe (TEU und Länge) in plausiblen Wertebereichen.

    Args:
        row_count: Anzahl der Datensätze
        rng: numpy Zufallsgenerator

    Returns:
        (load_teu, discharge_teu, length_m)
        - load_teu: Anzahl geladener TEU (int array)
        - discharge_teu: Anzahl zu entladender TEU (int array)
        - length_m: Schiffslänge in Metern (int array)

    Implementierung:
    - Werte werden als Integer-Arrays erzeugt (size=row_count).
    - Bereiche:
      - load_teu: 200–3000
      - discharge_teu: 150–2500
      - length_m: 250–400
    """
    load_teu = rng.integers(200, 3000, size=row_count)
    discharge_teu = rng.integers(150, 2500, size=row_count)
    length_m = rng.integers(250, 400, size=row_count)

    return load_teu, discharge_teu, length_m

# -------------------------------
# Hauptfunktion
# -------------------------------

def generate_carrierData(rowCount: int = 50, seed: int | None = None) -> pd.DataFrame:
    """
    Zweck:
    Erzeugt ein pandas.DataFrame mit synthetischen Carrier-/Schiffsdaten
    für Containerlogistik-Szenarien.

    Args:
        rowCount: Anzahl der zu generierenden Datensätze.
        seed: Optionaler Seed für reproduzierbare Zufallsdaten.

    Returns:
        pd.DataFrame mit den Spalten:
        - id: eindeutige Carrier-ID (laufende Zahlen)
        - shipName: Schiffsname (aus SHIP_NAMES)
        - linerName: Reedereiname (aus LINERS)
        - serviceName: Servicebezeichnung (passend zur Reederei)
        - serviceRoute: Route (passend zur Reederei)
        - eta: Estimated Time of Arrival (pd.Timestamp)
        - etd: Estimated Time of Departure (pd.Timestamp)
        - loadTEU: Anzahl geladener TEU
        - dischargeTEU: Anzahl entladener TEU
        - length_m: Schiffslänge in Metern

    Implementierungsdokumentation:
    1) RNG initialisieren:
       - np.random.default_rng(seed) für reproduzierbare Ergebnisse.

    2) IDs erzeugen:
       - carrier_ids = arange(1644600, 1644600 + rowCount)

    3) Reedereien zuweisen:
       - liner_ids werden zufällig aus LINERS.keys() gezogen.
       - liner_names wird über LINERS[liner_id] aufgelöst.

    4) Services passend zur Reederei:
       - pro liner_id wird _get_service_for_liner aufgerufen.
       - Ergebnis sind (serviceName, serviceRoute).

    5) Schiffsname ziehen:
       - rng.choice(SHIP_NAMES, size=rowCount, replace=True)

    6) Zeitpaare und Kennzahlen generieren:
       - ETA/ETD über _generate_ship_times
       - TEU/Länge über _generate_ship_metrics

    7) DataFrame erstellen:
       - Alle Arrays/Listen werden spaltenweise in pd.DataFrame zusammengeführt.
    """
    rng = np.random.default_rng(seed)

    carrier_ids = np.arange(1644600, 1644600 + rowCount, dtype=int)

    liner_ids = rng.choice(list(LINERS.keys()), size=rowCount, replace=True)
    liner_names = [LINERS[liner_id] for liner_id in liner_ids]

    service_data = [_get_service_for_liner(lid, rng) for lid in liner_ids]
    service_names, service_routes = zip(*service_data) if service_data else ([], [])

    ship_names = rng.choice(SHIP_NAMES, size=rowCount, replace=True)

    eta_list, etd_list = _generate_ship_times(rowCount, rng)
    load_teu, discharge_teu, length_m = _generate_ship_metrics(rowCount, rng)

    df = pd.DataFrame(
        {
            "id": carrier_ids,
            "shipName": ship_names,
            "linerName": liner_names,
            "serviceName": service_names,
            "serviceRoute": service_routes,
            "eta": eta_list,
            "etd": etd_list,
            "loadTEU": load_teu,
            "dischargeTEU": discharge_teu,
            "length_m": length_m,
        }
    )

    return df