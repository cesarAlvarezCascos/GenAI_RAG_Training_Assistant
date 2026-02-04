# src/export_topics.py
# Para ejecutar este script: python src/export_topics_csv.py

import os
import csv
import psycopg
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
conn = psycopg.connect(os.getenv("DATABASE_URL"))

def export_topics_csv(filename: str = "topics_export.csv"):
    """
    Exporta la tabla topics a un CSV listo para abrir en Excel.
    - Usa ';' como delimitador (típico en Excel en ES).
    - Guarda el archivo en la carpeta Descargas del usuario.
    """
    query = """
        SELECT
            id,
            model_topic_index,
            name,
            key_words,
            description
        FROM topics
        ORDER BY model_topic_index NULLS LAST, id;
    """

    with conn, conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
        colnames = [desc[0] for desc in cur.description]

    # Carpeta Descargas del usuario
    downloads_dir = Path.home() / "Downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    out_path = downloads_dir / filename

    print(f"[INFO] Exporting {len(rows)} topics to {out_path.resolve()}")

    # Abrimos el CSV con delimitador ';' para que Excel lo lea bien
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(
            f,
            delimiter=';',          # separador de campos
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL
        )
        # Cabecera
        writer.writerow(colnames)

        for row in rows:
            row_list = list(row)

            # Convertir arrays (key_words) a string "kw1, kw2, kw3"
            for i, val in enumerate(row_list):
                if isinstance(val, (list, tuple)):
                    row_list[i] = ", ".join(str(x) for x in val)
                elif val is None:
                    row_list[i] = ""

            writer.writerow(row_list)

    print("[INFO] Export complete.")

if __name__ == "__main__":
    export_topics_csv()
