import psycopg as pg


def connect_to_DB() -> pg.Connection:
    """
    Diese Methode wird verwendet um eine Verbindung zur DB herzustellen.
    
    :return: Gibt ein Connection-Objekt zurück mit Details zum verbinden auf die DB.
    :rtype: Connection[TupleRow]
    """
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