# Fake-Speicher (könnt ihr später durch DB ersetzen)
stored_fields = []

def save_fields(fields):
    global stored_fields
    stored_fields = fields

def get_fields():
    return stored_fields