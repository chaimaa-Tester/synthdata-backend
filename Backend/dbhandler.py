import psycopg as pg


def connect_to_DB() -> pg.Connection:
    try:
        conn = pg.connect(
            dbname='synthdatawizard',
            user='postgres',
            password='synthdata',
            host='localhost',
            port=5432
        )
        print("DB connection established succesfully!")
        return conn
    except pg.OperationalError as e:
        print(f"Couldnt establish DB connection, {e}")