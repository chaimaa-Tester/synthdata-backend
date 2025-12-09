from typing import List, Optional
from mimesis import *
from mimesis.enums import *
from faker import Faker
import random
import pandas as pd
import numpy as np
from field_schemas import FrontendField

person_gen = Person(Locale.DE)
address_gen = Address(Locale.DE)

rng = np.random.default_rng()

def generate_generalData(fields: List[FrontendField], num_rows: int, as_text_for_sheets: bool = False) -> pd.DataFrame:
    columns: dict[str, list] = {}

    # Felder trennen, erst unabhängig dann abhängig
    independent_fields = [f for f in fields if not (f.dependency or "").strip()]
    dependent_fields = [f for f in fields if (f.dependency or "").strip()]

    # 1. Unabhängige Felder generieren
    for field in independent_fields:
        ftype = (field.type or "").strip()
        dist_config = field.distributionConfig
        dist = (dist_config.distribution or "").strip().lower() if dist_config else ""

        # Primitive Datentypen
        if ftype == "string":
            # TODO: Wir brauche eine Implementierung für die eingabe eigener Werte aus den Tooltips
            data = [Faker.word() for _ in range(num_rows)]
        
        elif ftype == "number":
            data = [Faker.random_number() for _ in range(num_rows)]

        elif ftype == "boolean":
            data = [[True,False] for _ in range(num_rows)]

        elif ftype == "date":
            data = [Faker.date(pattern="dd.mm.yyyy") for _ in range(num_rows)]

        elif ftype == "time":
            data = [Faker.time(pattern="HH:MM:SS") for _ in range(num_rows)]

        elif ftype == "datetime":
            data = [Faker.date_time() for _ in range(num_rows)]

        # Personenbezogene Daten
        elif ftype == "firstname":
            data = [person_gen.first_name() for _ in range(num_rows)]
        
        elif ftype == "lastname":
            data = [person_gen.last_name() for _ in range(num_rows)]
        
        elif ftype == "fullname":
            data = [person_gen.full_name() for _ in range(num_rows)]

        elif ftype == "gender":
            data = [person_gen.gender() for _ in range(num_rows)]

        # Kommunikationsdaten
        elif ftype == "email":
            data = [person_gen.email(["web.de", "outlook.com", "yahoo.com", "gmail.com", "icloud.com"]) for _ in range(num_rows)]

        elif ftype == "telefon":
            data = [person_gen.phone_number() for _ in range(num_rows)]

        # Adressdaten
        elif ftype == "street":
            data = [address_gen.street_name() for _ in range(num_rows)]

        elif ftype == "house_number":
            data = [address_gen.street_number() for _ in range(num_rows)]

        elif ftype == "postcode":
            data = [address_gen.postal_code() for _ in range(num_rows)]
        
        elif ftype == "city":
            data = [address_gen.city() for _ in range(num_rows)]

        elif ftype == "state":
            data = [address_gen.state() for _ in range(num_rows)]

        elif ftype == "country":
            # TODO: Funktioniert das mit dem Locale ?
            data = [address_gen.country() for _ in range(num_rows)]
        
        elif ftype == "full_address":
            data = [address_gen.address() for _ in range(num_rows)]

        # Kategorien & Listen
        elif ftype == "enum":
            # TODO: Enum kann ein Typ übergeben werden für den das Enum erstellt werden soll!
            data = [Faker.enum() for _ in range(num_rows)]

        elif ftype == "list":
            data = []
            for _ in range(num_rows):
                data.append([Faker.word(), Faker.random_number()])

        # Musterbasierte Datentypen (Regex)

        # Identifikatoren
        elif ftype == "uuid":
            data = [Faker.uuid4() for _ in range(num_rows)]

        # Benutzerdefiniert
