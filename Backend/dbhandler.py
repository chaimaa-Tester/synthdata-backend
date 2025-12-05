import psycopg as pg


def connect_to_DB() -> pg.Connection:
    try:
        conn = pg.connect(
            dbname='synthdatawizard',
            user='postgres',
            password='synthdata',
            host='192.168.198.74',
            port=5432
        )
        print("DB connection established succesfully!")
        return conn
    except pg.OperationalError as e:
        print(f"Couldnt establish DB connection, {e}")