from typing import List
from faker import Faker
import random
import pandas as pd
from field_schemas import FrontendField

fake = Faker()

from typing import List
from faker import Faker
import random
import pandas as pd
from field_schemas import FrontendField

fake = Faker()

def generate_dummy_data(fields: List[FrontendField], num_rows: int = 10) -> pd.DataFrame:
    data = []

    for _ in range(num_rows):
        row = {}
        for field in fields:
            ftype = field.type.strip().lower()  # normalize input
            if ftype in ["string", "text"]:
                row[field.name] = fake.word()
            elif ftype == "int" or ftype == "integer":
                row[field.name] = random.randint(100, 9999)
            elif ftype == "float":
                row[field.name] = round(random.uniform(0, 1000), 2)
            elif ftype == "name":
                row[field.name] = fake.name()
            elif ftype == "email":
                row[field.name] = fake.email()
            elif ftype == "datetime":
                row[field.name] = fake.iso8601()
            elif ftype == "date":
                row[field.name] = fake.date()
            elif ftype == "time":
                row[field.name] = fake.time()
            elif ftype == "uuid":
                row[field.name] = fake.uuid4()
            elif ftype == "double":
                row[field.name] = round(random.uniform(0, 1000), 2)
            else:
                row[field.name] = "dummy"
        data.append(row)

    return pd.DataFrame(data)
