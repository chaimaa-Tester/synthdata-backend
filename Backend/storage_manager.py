import json
import uuid
import psycopg as pg

def _execute_query(conn, sql: str, params=None, fetch_one=False, fetch_all=False):
    """
    Hilfsmethode zum Ausführen von Queries und den Handling von Commit/ Rollback Aktionen.
    
    :param conn: Das Connection Objekt zum verbinden mit der DB.
    :param sql: Die SQL Abfrage welche an die DB geschickt wird.
    :type sql: str
    :param params: Die Parameter welche in die SQL Abfrage eingefügt werden. Default None.
    :param fetch_one: Wenn True wird aus dem Ergebnis der Query nur ein Wert gelesen.
    :param fetch_all: Wenn True werden alle Werte aus dem Ergebnis der Query gelesen.
    """
    try:
        # Cursor mit dem Verbindungsobjekt erstellen
        with conn.cursor() as cur:
            cur.execute(sql, params)

            if fetch_one:
                return cur.fetchone()
            if fetch_all:
                return cur.fetchall()

            # Änderungen committen, wenn es sich um INSERT/UPDATE/DELETE handelt
            conn.commit()
            print("Committed!")
    except pg.Error as e:
        conn.rollback()
        # Fehlerbehandlung: Ausgabe im Terminal
        print(f"Datenbankfehler bei Ausführung von '{sql[:50]}...': {e}")
        raise

def load_profiles(conn):
    """
    Lädt alle gespeicherten Profile aus der DB.
    
    :param conn: Das Connection Objekt der DB.
    """
    sql = "SELECT db_id, name FROM synthdata ORDER BY db_id DESC;"
    results = _execute_query(conn, sql, fetch_all=True)
    print(f"id={db_id}, name={name}" for db_id, name in results)
    conn.close()    

    if not results:
        return []
    profiles = [{"id": str(db_id), "name": name} for db_id, name in results]
    return profiles

# TODO: Implementierung der fehlenden Manager Methoden