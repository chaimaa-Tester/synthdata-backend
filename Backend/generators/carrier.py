# generator_carrier.py
from typing import List, Dict
import pandas as pd
import numpy as np

# -------------------------------
# Konfiguration - Reedereien
# -------------------------------
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

# Services nach Reederei gruppieren
SERVICES_BY_LINER: Dict[int, List[int]] = {}
for service_id, service_data in SERVICE_DETAILS.items():
    liner_id = service_data["linerId"]
    SERVICES_BY_LINER.setdefault(liner_id, []).append(service_id)

# -------------------------------
# Konfiguration - Schiffsnamen
# -------------------------------
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
    """Holt einen zufälligen Service für eine Reederei."""
    available_services = SERVICES_BY_LINER.get(liner_id, [])
    
    if available_services:
        service_id = rng.choice(available_services)
        service_data = SERVICE_DETAILS[service_id]
        return service_data["name"], service_data["route"]
    else:
        return "Generic Line", "Unbekannte Route"

def _generate_ship_times(row_count: int, rng: np.random.Generator) -> tuple[List[pd.Timestamp], List[pd.Timestamp]]:
    """Generiert realistische ETA/ETD Zeitpaare."""
    base_start = pd.Timestamp("2025-01-01")
    eta_list, etd_list = [], []
    
    for _ in range(row_count):
        # ETA: Zufällig in den nächsten 120 Tagen
        eta = base_start + pd.Timedelta(
            days=rng.integers(0, 120), 
            hours=rng.integers(0, 24)
        )
        
        # ETD: 12-120 Stunden nach ETA (realistische Liegezeit)
        dwell_hours = rng.integers(12, 120)
        etd = eta + pd.Timedelta(hours=dwell_hours)
        
        eta_list.append(eta)
        etd_list.append(etd)
    
    return eta_list, etd_list

def _generate_ship_metrics(row_count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generiert realistische Schiffsmetriken."""
    load_teu = rng.integers(200, 3000, size=row_count)      # Geladene Container
    discharge_teu = rng.integers(150, 2500, size=row_count) # Zu entladende Container
    length_m = rng.integers(250, 400, size=row_count)       # Schifflänge
    
    return load_teu, discharge_teu, length_m

# -------------------------------
# Hauptfunktion
# -------------------------------
def generate_carrierData(rowCount: int = 50, seed: int | None = None) -> pd.DataFrame:
    """
    Erzeugt synthetische Carrier-/Schiffsdaten für Containerlogistik.
    
    Args:
        rowCount: Anzahl der zu generierenden Carrier
        seed: Seed für reproduzierbare Zufallsdaten
    
    Returns:
        DataFrame mit Carrier-Daten:
        - id, shipName, linerName, serviceName, serviceRoute
        - eta, etd, loadTEU, dischargeTEU, length_m
    """
    rng = np.random.default_rng(seed)

    # Eindeutige Carrier-IDs
    carrier_ids = np.arange(1644600, 1644600 + rowCount, dtype=int)

    # Reedereien zufällig zuordnen
    liner_ids = rng.choice(list(LINERS.keys()), size=rowCount, replace=True)
    liner_names = [LINERS[liner_id] for liner_id in liner_ids]

    # Passende Services pro Reederei
    service_data = [_get_service_for_liner(lid, rng) for lid in liner_ids]
    service_names, service_routes = zip(*service_data) if service_data else ([], [])

    # Schiffsnamen
    ship_names = rng.choice(SHIP_NAMES, size=rowCount, replace=True)

    # Zeiten und Metriken generieren
    eta_list, etd_list = _generate_ship_times(rowCount, rng)
    load_teu, discharge_teu, length_m = _generate_ship_metrics(rowCount, rng)

    # DataFrame zusammenstellen
    df = pd.DataFrame({
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
    })

    return df