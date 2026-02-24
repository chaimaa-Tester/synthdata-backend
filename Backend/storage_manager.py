"""
--------------------------------------------------------------------
Projekt: SynthData Wizard
Modul: storage_manager.py
Autor: Burak Arabaci

Beschreibung:
Dieses Modul kapselt die JSON-basierte Persistenz der Profile.
Die Profile werden in einer lokalen Datei (data.json) gespeichert.

Funktionaler Umfang:
- Laden aller Profile aus data.json
- Speichern aller Profile nach data.json
- Hinzufügen eines neuen Profils (UUID als ID)
- Löschen eines Profils anhand seiner ID
- Speichern profilbezogener Daten (z.B. Konfigurationen)
- Abrufen profilbezogener Daten

Hinweis:
- Diese Persistenz ist bewusst leichtgewichtig (Dateispeicher statt DB),
  geeignet für Prototyp/Studentenprojekt.
- Für produktive Nutzung wären Locking/Concurrency-Handling und eine DB sinnvoll.
--------------------------------------------------------------------
"""

import json
import os
import uuid
from typing import Any, Dict, List


# Pfad zur JSON-Datei (liegt im selben Verzeichnis wie dieses Modul)
DATA_PATH = os.path.join(os.path.dirname(__file__), "data.json")


def load_profiles() -> List[Dict[str, Any]]:
    """
    Lädt alle gespeicherten Profile aus der Datei data.json.

    Robustheit:
    - Wenn die Datei nicht existiert, wird eine leere Liste zurückgegeben.
    - Wenn JSON ungültig ist oder ein Fehler beim Lesen auftritt,
      wird ebenfalls eine leere Liste zurückgegeben (Fail-Safe-Verhalten).

    Rückgabe:
    - Liste von Profil-Dictionaries, z.B. [{"id": "...", "name": "...", "data": {...}}, ...]
    """
    if not os.path.exists(DATA_PATH):
        return []

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            # Fail-Safe: bei korruptem JSON oder Lesefehlern nicht crashen
            return []


def save_profiles(profiles: List[Dict[str, Any]]) -> None:
    """
    Speichert alle Profile in die Datei data.json.

    Parameter:
    - profiles: vollständige Profil-Liste, die persistiert werden soll

    Designentscheidung:
    - Es wird immer die komplette Liste geschrieben (einfacher Ansatz).
      Für große Datenmengen wäre inkrementelles Speichern oder eine DB sinnvoll.
    """
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)


def add_profile(name: str) -> Dict[str, Any]:
    """
    Fügt ein neues Profil hinzu und persistiert es direkt.

    Ablauf:
    1) Bestehende Profile laden
    2) Neues Profilobjekt erzeugen (UUID als eindeutige ID)
    3) Profil anhängen und Gesamtdatei speichern

    Parameter:
    - name: Profilname (validiert i.d.R. im API-Layer)

    Rückgabe:
    - das neu erstellte Profilobjekt
    """
    profiles = load_profiles()
    new_profile = {"id": str(uuid.uuid4()), "name": name}

    profiles.append(new_profile)
    save_profiles(profiles)

    return new_profile


def delete_profile(profile_id: str) -> None:
    """
    Löscht ein Profil anhand seiner ID.

    Parameter:
    - profile_id: ID des zu löschenden Profils
    """
    profiles = load_profiles()
    profiles = [p for p in profiles if p.get("id") != profile_id]
    save_profiles(profiles)


def save_profile_data(profile_id: str, data: Dict[str, Any]) -> None:
    """
    Speichert zusätzliche Daten innerhalb eines bestimmten Profils.

    Zweck:
    - Pro Profil können kontextbezogene Konfigurationen gespeichert werden
      (z.B. Generator-Einstellungen, Felddefinitionen, UI-State etc.)

    Parameter:
    - profile_id: Zielprofil-ID
    - data: beliebige strukturierte Daten (Dictionary)

    Hinweis:
    - Falls profile_id nicht existiert, wird still gespeichert, ohne Änderung.
      (Optional: könnte man hier bewusst einen Fehler werfen)
    """
    profiles = load_profiles()

    for profile in profiles:
        if profile.get("id") == profile_id:
            profile["data"] = data
            break

    save_profiles(profiles)


def get_profile_data(profile_id: str) -> Dict[str, Any]:
    """
    Lädt gespeicherte Daten eines bestimmten Profils.

    Verhalten:
    - Wenn Profil existiert: Rückgabe von profile["data"] oder {} (Fallback)
    - Wenn Profil nicht existiert: {} (leeres Objekt)

    Parameter:
    - profile_id: ID des gewünschten Profils

    Rückgabe:
    - Dictionary mit gespeicherten Profil-Daten
    """
    profiles = load_profiles()

    for profile in profiles:
        if profile.get("id") == profile_id:
            return profile.get("data", {})

    return {}