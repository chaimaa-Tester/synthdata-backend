# backend/api/name_source.py
from fastapi import APIRouter, HTTPException
from mimesis import Person
from mimesis.locales import Locale
from mimesis.enums import Gender
from typing import List, Dict, Optional

router = APIRouter(prefix="/api/name-source", tags=["name-generation"])

# Mapping von Frontend-Ländern zu mimesis Locales
COUNTRY_TO_LOCALE = {
    "Deutschland": Locale.DE,
    "Österreich": Locale.DE,
    "Schweiz": Locale.DE,
    "Türkei": Locale.TR,
    "Spanien": Locale.ES,
    "USA": Locale.EN,
    "Frankreich": Locale.FR,
    "Italien": Locale.IT,
    "Brasilien": Locale.PT_BR,
    "Japan": Locale.JA,
    "China": Locale.ZH,
    "Russland": Locale.RU,
}

@router.post("/generate")
async def generate_names(
    source_type: str,           # "western" oder "regional"
    country: Optional[str] = None,  # Nur bei "regional"
    name_field_type: str = "full",  # "vorname", "nachname", "name"
    count: int = 10,
    gender: Optional[str] = None   # "male", "female", None für zufällig
):
    """
    Generiert Namen basierend auf Quelle und Land.
    
    Args:
        source_type: "western" für westliche Namen, "regional" für länderspezifisch
        country: Landname (nur bei regional)
        name_field_type: Art des Namens ("vorname", "nachname", "name")
        count: Anzahl zu generierender Namen
        gender: Geschlecht ("male", "female") oder None für gemischt
    """
    try:
        # 1. Locale bestimmen
        if source_type == "western":
            locale = Locale.EN  # Englisch als Standard "westlich"
        elif source_type == "regional" and country:
            locale = COUNTRY_TO_LOCALE.get(country, Locale.EN)
            if locale is None:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Land '{country}' wird nicht unterstützt"
                )
        else:
            locale = Locale.EN
        
        # 2. Geschlecht für mimesis setzen
        mimesis_gender = None
        if gender == "male":
            mimesis_gender = Gender.MALE
        elif gender == "female":
            mimesis_gender = Gender.FEMALE
        
        # 3. Person-Generator erstellen
        person = Person(locale)
        
        # 4. Namen generieren
        generated_names = []
        
        for _ in range(count):
            if name_field_type == "vorname":
                # Nur Vorname
                if mimesis_gender:
                    name = person.first_name(gender=mimesis_gender)
                else:
                    # Zufälliges Geschlecht für jeden Namen
                    random_gender = Gender.MALE if person.random.randint(0, 1) == 0 else Gender.FEMALE
                    name = person.first_name(gender=random_gender)
                    
            elif name_field_type == "nachname":
                # Nur Nachname (kein Geschlecht bei Nachnamen in mimesis)
                name = person.last_name()
                
            else:  # "name" oder "full"
                # Vollständiger Name
                if mimesis_gender:
                    first = person.first_name(gender=mimesis_gender)
                    last = person.last_name()
                else:
                    # Zufälliges Geschlecht
                    random_gender = Gender.MALE if person.random.randint(0, 1) == 0 else Gender.FEMALE
                    first = person.first_name(gender=random_gender)
                    last = person.last_name()
                name = f"{first} {last}"
            
            generated_names.append(name)
        
        return {
            "success": True,
            "source_type": source_type,
            "country": country,
            "locale": str(locale),
            "name_field_type": name_field_type,
            "count": len(generated_names),
            "names": generated_names
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/supported-countries")
async def get_supported_countries():
    """Gibt unterstützte Länder zurück"""
    return {
        "supported_countries": list(COUNTRY_TO_LOCALE.keys())
    }