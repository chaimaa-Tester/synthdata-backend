"""
--------------------------------------------------------------------
Projekt: SynthData Wizard
Datei: backend/field_schemas.py
Autor: Burak Arabaci


Beschreibung:
Dieses Modul enthält die Pydantic-Modelle (Schemas), die als Datentransfer-
Objekte (DTOs) zwischen Frontend und Backend dienen.

Zweck:
- Validierung eingehender Requests (z.B. /api/export)
- Einheitliche Typdefinition für Feld- und Export-Konfigurationen
- Klare Contract-Schnittstelle zwischen UI-Konfiguration und Daten-Generatoren

Hinweis:
- Die Modelle sind bewusst flexibel gehalten, da Felder dynamisch im Frontend
  konfiguriert werden können (z.B. DistributionConfig, customValues, valueSource).
--------------------------------------------------------------------
"""

from pydantic import BaseModel
from typing import List, Optional


class FieldDefinition(BaseModel):
    """
    Allgemeines Feldmodell zur Beschreibung einer Spalte/Felddefinition.

    Verwendung:
    - Kann z.B. für interne Definitionen oder generische Feldlisten genutzt werden.

    Attribute:
    - name: Anzeigename / Spaltenname
    - type: Feldtyp (z.B. number, text, vorname, nachname, ...)
    - dependency: Optionaler Verweis auf anderes Feld (z.B. abhängige Generierung)
    - DoNotShowinTable: UI-Hinweis, ob das Feld in Tabellenansichten ausgeblendet wird
    """
    name: str
    type: str
    dependency: Optional[str] = None
    DoNotShowinTable: Optional[bool] = False


class DistributionConfig(BaseModel):
    """
    Konfiguration für die statistische Verteilung eines Feldes.

    Zweck:
    - Wird genutzt, um numerische Werte anhand einer Verteilung zu generieren
      oder um erkannte Verteilungen (z.B. via FileUploadModal) zu persistieren.

    Attribute:
    - distribution: Name der Verteilung (z.B. "norm", "gamma", "lognorm", ...)
    - parameterA / parameterB: Hauptparameter (je nach Verteilung z.B. mean/std)
    - extraParams: optionale Zusatzparameter (für Verteilungen mit mehr Parametern)
    """
    distribution: Optional[str] = None
    parameterA: Optional[str] = None
    parameterB: Optional[str] = None
    extraParams: Optional[List[str]] = None


class FrontendField(BaseModel):
    """
    Feldmodell, wie es vom Frontend an das Backend gesendet wird (z.B. /api/export).

    Motivation:
    - Das Frontend konfiguriert dynamisch die Feldliste (rows)
    - Das Backend nutzt diese Informationen zur Datengenerierung

    Attribute:
    - name: Feld-/Spaltenname (muss später mit df.columns matchen)
    - type: Feldtyp (steuert Generator-Logik)
    - dependency: optionaler Bezug zu einem anderen Feld
    - distributionConfig: optionale Konfiguration für numerische Verteilungen
    - valueSource: Quelle der Werte (z.B. "custom", "generated", "file", ...)
    - customValues: Liste benutzerdefinierter Werte (z.B. kategoriale Ausprägungen)
    - nameSource: optionale Namensquelle (z.B. "western" oder "regional")
    """
    name: str
    type: str
    dependency: Optional[str] = None
    distributionConfig: Optional[DistributionConfig] = None
    valueSource: Optional[str] = None
    customValues: Optional[List[str]] = None
    nameSource: Optional[str] = None


class ExportSheet(BaseModel):
    """
    Modell für die Excel-Sheet-Konfiguration (XLSX-Export).

    Zweck:
    - Das Frontend kann mehrere Sheets definieren und pro Sheet festlegen,
      welche Felder (Spalten) exportiert werden sollen.

    Attribute:
    - id: Sheet-Identifikator (Frontend-intern)
    - name: Sheet-Name (wird später im Backend validiert/gesäubert)
    - fieldNames: Liste der Feldnamen, die in dieses Sheet exportiert werden
    - locked: optionaler UI-/Logik-Flag (z.B. "Sheet ist gesperrt")
    """
    id: str
    name: str
    fieldNames: List[str]
    locked: Optional[bool] = False


class ExportRequest(BaseModel):
    """
    Request-Modell für den Export-Endpunkt (/api/export).

    Enthält:
    - rows: Liste der konfigurierten Felder (FrontendField)
    - rowCount: Anzahl zu generierender Datensätze
    - format: Exportformat (CSV, XLSX, JSON, SQL, ...)
    - lineEnding: Zeilenende-Konfiguration (relevant bei CSV)
    - usedUseCaseIds: Liste ausgewählter Use-Cases (steuert Generator-Auswahl)
    - sheets: optionale Sheet-Konfiguration (nur relevant für XLSX)

    Hinweis:
    - sheets ist optional, da XLSX-Export auch ohne Sheet-Aufteilung funktionieren soll
      (Fallback: ein Sheet mit allen Spalten).
    """
    rows: List[FrontendField]
    rowCount: int
    format: str
    lineEnding: str
    usedUseCaseIds: List[str]
    sheets: Optional[List[ExportSheet]] = None