# src/export_feedback_to_csv.py
# Ejecutando el comando python src/feedback-csv/export_feedback_csv.py descargara un archivo feedback_unverified.csv

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

def export_unverified_feedback():
    downloads_dir = Path.home() / "Downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    output_path = downloads_dir / "feedback_unverified.csv"

    result = (
        supabase.table("feedback")
        .select("*")
        .eq("is_verified", False)
        .order("created_at", desc=True)
        .execute()
    )

    rows = result.data or []
    if not rows:
        print("No hay feedback no verificado para exportar.")
        return

    df = pd.DataFrame(rows)

    wanted_cols = [
        "id", "created_at", "user_id", "session_id",
        "query", "answer", "rating", "feedback_type",
        "comment", "is_verified"
    ]
    df = df[[c for c in wanted_cols if c in df.columns]]

    # 🔥 EXPORTACIÓN 100% COMPATIBLE EXCEL
    df.to_csv(
        output_path,
        index=False,
        sep=";",                  # Excel lo lee perfecto
        encoding="utf-8-sig",     # Excel reconoce UTF-8 con BOM
        quoting=csv.QUOTE_ALL     # Evita que las comas rompan columnas
    )

    print(f"✅ Exportado correctamente a → {output_path}")

if __name__ == "__main__":
    export_unverified_feedback()
