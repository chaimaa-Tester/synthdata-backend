from typing import List
from faker import Faker
from scipy.stats import norm, uniform, gamma, poisson
import random
import pandas as pd
from field_schemas import FrontendField, DistributionConfig


# fake = Faker()

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
        dist = field.distributionConfig.distribution.lower()
        paramA = field.distributionConfig.parameterA
        paramB = field.distributionConfig.parameterB

        if dist == "normal":
            mu = float(paramA or 0)
            sigma = float(paramB or 1)
            data = norm.rvs(loc=mu, scale=sigma, size=num_rows)

        elif dist == "gamma":
            shape = float(paramA or 1)
            scale = float(paramB or 1)
            data = gamma.rvs(a=shape, scale=scale, size=num_rows)

        elif dist == "uniform":
            low = float(paramA or 0)
            high = float(paramB or 1)
            data = uniform.rvs(loc=low, scale=high - low, size=num_rows)


        else:
            # Fallback für Strings
            data = [f"{field.name}_{i}" for i in range(num_rows)]

        df[field.name] = data

    return df