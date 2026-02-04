# src/import_feedback_csv.py
# Ejecutando el comando python src/feedback-csv/import_feedback_csv.py se importara el archivo desde la misma ruta en la que se descargo y se haran los respectivos cambios en la base de datos.

import os
from pathlib import Path
import pandas as pd
import csv
from dotenv import load_dotenv
from supabase import create_client

project_root = Path(__file__).resolve().parents[2]
load_dotenv(project_root / ".env")

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

def to_bool(value):
    """Convierte valores típicos del CSV a booleano real."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    
    s = str(value).strip().lower()
    return s in ("true", "1", "yes", "y", "si", "sí")

def import_verified_feedback():
    # Ruta al CSV exportado
    downloads_dir = Path.home() / "Downloads"
    csv_path = downloads_dir / "feedback_unverified.csv"

    if not csv_path.exists():
        print(f"❌ No se encontró el archivo: {csv_path}")
        return

    print(f"📥 Leyendo CSV: {csv_path}")

    # 🔥 MISMA CONFIGURACIÓN QUE EL EXPORT
    try:
        df = pd.read_csv(
            csv_path,
            sep=";",                # separador usado en exportación
            encoding="utf-8-sig",    # UTF-8 con BOM
            quotechar='"',           # respeta las comillas de Excel
            dtype=str                # evitar que pandas convierta números automáticamente
        )
    except Exception as e:
        print("❌ Error leyendo CSV:", e)
        return

    if "id" not in df.columns or "is_verified" not in df.columns:
        print("❌ El CSV debe contener las columnas 'id' e 'is_verified'.")
        return

    # Convertir columna is_verified a booleano real
    df["is_verified"] = df["is_verified"].apply(to_bool)

    # Filtrar solo los que el experto marcó como true
    to_update = df[df["is_verified"] == True]

    if to_update.empty:
        print("ℹ️ No hay filas marcadas como verificadas.")
        return

    print(f"🔎 Filas a actualizar en la BD: {len(to_update)}\n")

    updated_count = 0

    # Actualizar una por una en Supabase
    for _, row in to_update.iterrows():
        fb_id = row["id"]

        print(f"→ Actualizando feedback id={fb_id} → is_verified = true")

        supabase.table("feedback")\
            .update({"is_verified": True})\
            .eq("id", fb_id)\
            .execute()

        updated_count += 1

    print(f"\n🎯 Actualización completada: {updated_count} filas actualizadas.")

if __name__ == "__main__":
    import_verified_feedback()
