"""
Autor: Jan Krämer

Die Methode implementiert die Funktionalität zum Verbindungsaufbau mit der PostgreSQL Datenbank.
Sie wird von der storage_manager.py verwendet um den Methoden ein Verbindungsobjekt zu übergeben.
"""

import psycopg as pg

def connect_to_DB() -> pg.Connection:
    """
    Diese Methode wird verwendet um eine Verbindung zur DB herzustellen. Danach wird ein Connection Objekt zurück gegeben
    welches alle Informationen enthält, mit denen sich andere Methoden zur DB verbinden können.
    
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