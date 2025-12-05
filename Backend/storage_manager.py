import psycopg as pg
import json
import uuid

def _execute_query(conn, sql, params=None, fetch_one=False, fetch_all=False):
    """Führt eine Query aus und behandelt Commit/Rollback."""
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

"""Lädt alle gespeicherten Profile."""
def load_profiles(conn):
    sql = "SELECT db_id, name FROM synthdata ORDER BY db_id DESC;"
    results = _execute_query(conn, sql, fetch_all=True)
    print(f"id={db_id}, name={name}" for db_id, name in results)
    conn.close()

    if not results:
        return []
    
    profiles = [{"id": str(db_id), "name": name} for db_id, name in results]        
    return profiles

"""Fügt ein neues Profil hinzu."""
def add_profile(conn, name):
    sql = """
        INSERT INTO synthdata (name, row_count, format, line_ending, rows_data)
        VALUES (%s,%s,%s,%s,%s) RETURNING db_id;
    """

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

"""Löscht ein Profil anhand seiner ID."""
def delete_profile(conn, db_id):
    sql = "DELETE FROM synthdata WHERE db_id = %s;"
    _execute_query(conn, sql, (db_id,))

"""Speichert Daten innerhalb eines bestimmten Profils."""
def save_profile_data(conn, db_id, data):
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

"""Lädt gespeicherte Daten eines bestimmten Profils."""
def get_profile_data(conn, db_id):
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