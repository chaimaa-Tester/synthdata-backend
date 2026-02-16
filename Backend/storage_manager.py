import json
import psycopg as pg

# === Diese Klasse wurde erstellt von Burak Arabaci. Überarbeitet von Jan Krämer zur Verbindung auf die DB. ===

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

def add_profile(conn, name):
    """
    Fügt ein neues Profil in die DB.
    
    :param conn: Das Connection Objekt der DB.
    :param name: Name des erstellten Profils.
    """
    sql = """
        INSERT INTO synthdata (name, row_count, format, line_ending, rows_data)
        VALUES (%s,%s,%s,%s,%s) RETURNING db_id;
    """

    # Die beigefügten Daten sind Standarddaten aus der UI.
    params = (
        name,
        10,
        "CSV",
        "Windows(CRLF)",
        json.dumps([])
    )

    result= _execute_query(conn, sql, params, fetch_one=True)
    conn.commit()
    print("Committed!")
    conn.close()

    if result:
        new_id = result[0]
        print(f"id={new_id}, name={name}")
        return {"id": str(new_id), "name": name}
    return None

def delete_profile(conn, db_id):
    """
    Löscht das angegebene Profil aus der DB anhand seiner ID.
    
    :param conn: Das Connection Objekt der DB.
    :param db_id: Die ID des Profils in der DB.
    """
    sql = "DELETE FROM synthdata WHERE db_id = %s;"
    _execute_query(conn, sql, (db_id,))

def save_profile_data(conn, db_id, data):
    """
    Speichert die Daten innerhalb eines bestimmten Profils.
    
    :param conn: Das Connection Objekt der DB.
    :param db_id: Die ID des Profils in der DB.
    :param data: Die Daten aus der UI welche per JSON Dump in der DB gespeichert werden.
    """
    rows_data = data.get("rows", [])
    print(rows_data)
    row_count = data.get("rowCount", 10)
    print(row_count)
    fmt = data.get("format", "CSV")
    print(fmt)
    line_ending = data.get("lineEnding", "Windows(CRLF)")
    print(line_ending)
    print(db_id)

    sql = """
        UPDATE synthdata
        SET rows_data = %s, row_count = %s, format = %s, line_ending = %s
        WHERE db_id = %s;
    """
    params = (
        json.dumps(rows_data),
        row_count,
        fmt,
        line_ending,
        db_id
    )

    _execute_query(conn, sql, params)
    conn.close()

def get_profile_data(conn, db_id):
    """
    Lädt gespeicherte Daten des angegebenen Profils.
    
    :param conn: Das Connection Objekt der DB.
    :param db_id: Die ID des Profils in der DB.
    """
    sql = """
        SELECT rows_data, row_count, format, line_ending
        FROM synthdata
        WHERE db_id = %s;
    """
    result = _execute_query(conn, sql, (db_id,), fetch_one=True)    
    conn.close()
    print(result)

    if result:
        rows_data, row_count, fmt, line_ending = result

        return {
            "rows": rows_data,
            "rowCount": row_count,
            "format": fmt,
            "lineEnding": line_ending
        }
    return {}