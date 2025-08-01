from typing import List
from faker import Faker
from scipy.stats import norm, uniform, gamma, poisson
import random
import pandas as pd
from field_schemas import FrontendField, DistributionConfig


fake = Faker()

# def generate_dummy_data(fields: List[FrontendField], num_rows: int = 10) -> pd.DataFrame:
#     data = []

#     for _ in range(num_rows):
#         row = {}
#         for field in fields:
#             ftype = field.type.strip().lower()  # normalize input
#             if ftype in ["string", "text"]:
#                 row[field.name] = fake.word()
#             elif ftype == "int" or ftype == "integer":
#                 row[field.name] = random.randint(100, 9999)
#             elif ftype == "float":
#                 row[field.name] = round(random.uniform(0, 1000), 2)
#             elif ftype == "name":
#                 row[field.name] = fake.name()
#             elif ftype == "email":
#                 row[field.name] = fake.email()
#             elif ftype == "datetime":
#                 row[field.name] = fake.iso8601()
#             elif ftype == "date":
#                 row[field.name] = fake.date()
#             elif ftype == "time":
#                 row[field.name] = fake.time()
#             elif ftype == "uuid":
#                 row[field.name] = fake.uuid4()
#             elif ftype == "double":
#                 row[field.name] = round(random.uniform(0, 1000), 2)
#             else:
#                 row[field.name] = "dummy"
#         data.append(row)

#     return pd.DataFrame(data)


def generate_dummy_data(fields: List[FrontendField], num_rows: int) -> pd.DataFrame:
    df = pd.DataFrame()

    for field in fields:
        ftype = field.type.strip().lower()  # normalize input
        dist_config = field.distributionConfig
        dist = dist_config.distribution.lower() if dist_config else None


       
        if ftype in ["string", "text"]:
            # Fallback für Strings
            data = [fake.word() for _ in range(num_rows)]

        # Für Double Typen
        elif ftype == "double":
            paramA = float(dist_config.parameterA) if dist_config and dist_config.parameterA else None
            paramB = float(dist_config.parameterB) if dist_config and dist_config.parameterB else None
            if dist == "normal":
                mu = paramA or 0
                sigma = paramB or 1
                data = norm.rvs(loc=mu, scale=sigma, size=num_rows)
            elif dist == "uniform":
                low = paramA or 0
                high = paramB or 100
                data = uniform.rvs(loc=low, scale=high - low, size=num_rows)
            elif dist == "gamma":
                shape = paramA or 1
                scale = paramB or 1
                data = gamma.rvs(a=shape, scale=scale, size=num_rows)    
            else:
                data = [round(random.uniform(0, 1000), 2) for _ in range(num_rows)]
        
        elif ftype == "date":
            paramA = (dist_config.parameterA) if dist_config and dist_config.parameterA else None
            paramB = (dist_config.parameterB) if dist_config and dist_config.parameterB else None
            if dist == "uniform":
                # Dynamische Prüfung und Konvertierung der Parameter
                start_date = pd.to_datetime(paramA or "2000-01-01")
                end_date = pd.to_datetime(paramB or "2025-01-01")
                # Generiere zufällige Datumswerte zwischen den Grenzen
                data = [
                    fake.date_between(start_date=start_date, end_date=end_date).isoformat()
                    for _ in range(num_rows)
        ]
            else:
                # Fallback für Datum
                data = [fake.date() for _ in range(num_rows)]      



        else:
            # Fallback für Strings
            data = [f"{field.name}_{i}" for i in range(num_rows)]

        df[field.name] = data

    return df