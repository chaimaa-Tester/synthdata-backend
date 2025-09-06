from typing import List
from faker import Faker
from scipy.stats import norm, uniform, gamma, poisson, binom
import random
import pandas as pd
from field_schemas import FrontendField, DistributionConfig
import numpy as np

fake = Faker()

def generate_dummy_data(fields: List[FrontendField], num_rows: int) -> pd.DataFrame:
    df = pd.DataFrame()

    for field in fields:
        ftype = field.type.strip().lower()  # normalize input
        dist_config = field.distributionConfig
        dist = dist_config.distribution.lower() if dist_config and dist_config.distribution else None

        if ftype in ["string", "text"]:
            # Für String Typen
            paramA = dist_config.parameterA if dist_config and dist_config.parameterA else None
            paramB = dist_config.parameterB if dist_config and dist_config.parameterB else None
            
            if dist == "uniform":
                # Gleichverteilung aus einer Liste von Werten
                values = [v.strip() for v in paramA.split(',')] if paramA else [fake.word() for _ in range(5)]
                data = [random.choice(values) for _ in range(num_rows)]
            elif dist == "categorical":
                # Kategoriale Verteilung mit Gewichten
                values = [v.strip() for v in paramA.split(',')] if paramA else [fake.word() for _ in range(5)]
                weights = [float(w.strip()) for w in paramB.split(',')] if paramB else [1.0] * len(values)
                
                # Normalisiere Gewichte
                total = sum(weights)
                norm_weights = [w/total for w in weights]
                data = np.random.choice(values, size=num_rows, p=norm_weights)
            else:
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
            elif dist == "lognormal":
                mu = paramA or 0
                sigma = paramB or 1
                data = np.random.lognormal(mean=mu, sigma=sigma, size=num_rows)
            elif dist == "exponential":
                rate = paramA or 1
                data = np.random.exponential(scale=1/rate, size=num_rows)
            else:
                data = [round(random.uniform(0, 1000), 2) for _ in range(num_rows)]
        
        # Für Integer Typen
        elif ftype in ["int", "integer"]:
            paramA = float(dist_config.parameterA) if dist_config and dist_config.parameterA else None
            paramB = float(dist_config.parameterB) if dist_config and dist_config.parameterB else None
            
            if dist == "uniform":
                low = int(paramA) if paramA is not None else 0
                high = int(paramB) if paramB is not None else 100
                data = [random.randint(low, high) for _ in range(num_rows)]
            elif dist == "normal":
                mu = paramA or 0
                sigma = paramB or 1
                # Erzeuge normale Verteilung und runde auf ganze Zahlen
                data = np.round(norm.rvs(loc=mu, scale=sigma, size=num_rows)).astype(int)
            elif dist == "binomial": #'n' muss ≥ 0 und 'p' zwischen 0 und 1 sein.
                n = int(paramA) if paramA is not None else 10
                p = paramB or 0.5
                if n < 0:
                    raise ValueError("'n' muss ≥ 0 sein.")
                if not (0 <= p <= 1):
                    raise ValueError("'p' muss zwischen 0 und 1 sein.")
                data = binom.rvs(n=n, p=p, size=num_rows)
            elif dist == "poisson":
                lam = paramA or 1
                data = poisson.rvs(mu=lam, size=num_rows)
            else:
                data = [random.randint(100, 9999) for _ in range(num_rows)]
        
        elif ftype == "date":
            paramA = dist_config.parameterA if dist_config and dist_config.parameterA else None
            paramB = dist_config.parameterB if dist_config and dist_config.parameterB else None
            
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
            # Fallback für unbekannte Typen
            data = [f"{field.name}_{i}" for i in range(num_rows)]

        df[field.name] = data

    return df