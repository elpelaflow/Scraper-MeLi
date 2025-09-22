"""Utility script to backfill history tables before enabling the new snapshot flow."""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime
from pathlib import Path

from data_transformation import archive_previous_snapshot


def _resolve_snapshot_label(
    conn: sqlite3.Connection, provided: str | None
) -> str:
    if provided:
        return provided

    row = conn.execute("SELECT MAX(scrap_date) FROM mercadolivre_items").fetchone()
    if row and row[0]:
        return str(row[0])

    return datetime.now().isoformat()


def migrate(database_path: str | Path, snapshot_date: str | None = None) -> None:
    db_path = Path(database_path)
    if not db_path.exists():
        print(f"No se encontró la base de datos en {db_path}.")
        return

    with sqlite3.connect(db_path) as conn:
        label = _resolve_snapshot_label(conn, snapshot_date)
        archived = archive_previous_snapshot(conn, label)
        if archived:
            conn.commit()
            print(
                "Histórico migrado correctamente. snapshot_date utilizado: "
                f"{label}"
            )
        else:
            print(
                "No se encontraron registros vigentes para copiar al historial."
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Crea las tablas *_history y copia el contenido actual "
            "etiquetándolo con un snapshot_date para preservar las corridas previas."
        )
    )
    parser.add_argument(
        "--database",
        default=str(Path(__file__).resolve().parent.parent / "data" / "database.db"),
        help="Ruta al archivo SQLite que se debe migrar.",
    )
    parser.add_argument(
        "--snapshot-date",
        dest="snapshot_date",
        default=None,
        help=(
            "Valor opcional para usar como snapshot_date. Si no se provee se utiliza "
            "el scrap_date más reciente detectado o el timestamp actual."
        ),
    )

    args = parser.parse_args()
    migrate(args.database, args.snapshot_date)


if __name__ == "__main__":
    main()
