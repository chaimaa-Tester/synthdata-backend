import json
import os
import uuid

DATA_PATH = os.path.join(os.path.dirname(__file__), "data.json")

"""Lädt alle gespeicherten Profile."""
def load_profiles():
    if not os.path.exists(DATA_PATH):
        return []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return []

"""Speichert alle Profile in data.json."""
def save_profiles(profiles):
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)

"""Fügt ein neues Profil hinzu."""
def add_profile(name):
    profiles = load_profiles()
    new_profile = {"id": str(uuid.uuid4()), "name": name}
    profiles.append(new_profile)
    save_profiles(profiles)
    return new_profile

"""Löscht ein Profil anhand seiner ID."""
def delete_profile(id):
    profiles = load_profiles()
    profiles = [p for p in profiles if p.get("id") != id]
    save_profiles(profiles)

"""Speichert Daten innerhalb eines bestimmten Profils."""
def save_profile_data(profile_id, data):
    profiles = load_profiles()
    for profile in profiles:
        if profile.get("id") == profile_id:
            profile["data"] = data
            break
    save_profiles(profiles)

"""Lädt gespeicherte Daten eines bestimmten Profils."""
def get_profile_data(profile_id):
    profiles = load_profiles()
    for profile in profiles:
        if profile.get("id") == profile_id:
            return profile.get("data", {})
    return {}