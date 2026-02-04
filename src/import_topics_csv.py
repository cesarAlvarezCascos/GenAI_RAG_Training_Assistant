# src/import_topics_csv.py
import os
import csv
import psycopg
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
conn = psycopg.connect(os.getenv("DATABASE_URL"))


def _parse_keywords_cell(val: str | None):
    """
    Convierte 'kw1, kw2, kw3' -> ['kw1','kw2','kw3']
    Si está vacío, devuelve None (para guardar NULL).
    """
    if not val:
        return None
    parts = [p.strip() for p in val.split(",")]
    kws = [p for p in parts if p]
    return kws or None


def import_topics_csv(filename: str = "topics_export.csv"):
    """
    Importa cambios desde el CSV de Descargas a la tabla topics.

    Comportamiento:
    - Si la fila tiene 'id' -> UPDATE de esa fila en topics.
    - Si la fila no tiene 'id' pero tiene algún contenido -> INSERT en topics.
    - Sobrescribe siempre model_topic_index, name, key_words y description
      con lo que venga en el CSV (vacío -> NULL).
    """
    downloads_dir = Path.home() / "Downloads"
    in_path = downloads_dir / filename

    if not in_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {in_path}")

    print(f"[INFO] Importing topics from {in_path.resolve()}")

    updated = 0
    inserted = 0
    skipped = 0

    with in_path.open("r", encoding="utf-8") as f, conn, conn.cursor() as cur:
        reader = csv.DictReader(f, delimiter=';', quotechar='"')

        for row in reader:
            row_id = (row.get("id") or "").strip()
            model_idx_raw = (row.get("model_topic_index") or "").strip()
            name_raw = (row.get("name") or "").strip()
            kw_raw = (row.get("key_words") or "").strip()
            description_raw = (row.get("description") or "").strip()

            # Parseos
            # model_topic_index: int o None
            model_idx = None
            if model_idx_raw:
                try:
                    model_idx = int(model_idx_raw)
                except ValueError:
                    model_idx = None

            name = name_raw or None
            key_words = _parse_keywords_cell(kw_raw)
            description = description_raw or None

            # Fila vacía (sin id y sin contenido relevante) -> skip
            if not row_id and not (model_idx is not None or name or key_words or description):
                skipped += 1
                continue

            # Caso 1: fila existente -> UPDATE
            if row_id:
                cur.execute(
                    """
                    UPDATE topics
                    SET
                        model_topic_index = %s,
                        name = %s,
                        key_words = %s,
                        description = %s
                    WHERE id = %s;
                    """,
                    (model_idx, name, key_words, description, row_id)
                )
                updated += 1
                continue

            # Caso 2: fila nueva (sin id) -> INSERT
            cur.execute(
                """
                INSERT INTO topics (model_topic_index, name, key_words, description)
                VALUES (%s, %s, %s, %s);
                """,
                (model_idx, name, key_words, description)
            )
            inserted += 1

    print(f"[INFO] Import finished. Updated: {updated}, Inserted: {inserted}, Skipped: {skipped}")


if __name__ == "__main__":
    import_topics_csv()
