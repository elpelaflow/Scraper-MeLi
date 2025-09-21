+268
-44

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd


LISTING_OPTIONAL_COLUMNS = ("product_url", "item_id")
DETAIL_COLUMNS = (
    "item_id",
    "product_url",
    "title_detail",
    "price_detail",
    "currency",
    "condition",
    "sold_quantity",
    "brand",
    "model",
    "color",
    "material",
    "capacity",
    "voltage",
    "seller_name",
    "seller_location",
    "official_store_flag",
    "shipping_full_flag",
    "shipping_free_flag",
    "rating_value",
    "rating_count",
    "description_plain",
    "images_json",
    "breadcrumbs_json",
    "warranty_text",
    "returns_text",
    "scrap_date_detail",
)

BOOL_COLUMNS = ("official_store_flag", "shipping_full_flag", "shipping_free_flag")
INT_COLUMNS = ("sold_quantity", "rating_count")
FLOAT_COLUMNS = ("rating_value",)
JSON_COLUMNS = ("images_json", "breadcrumbs_json")


def read_json_dataframe(path: str | Path) -> pd.DataFrame:
    try:
        return pd.read_json(path)
    except ValueError:
        return pd.DataFrame()
    except FileNotFoundError:
        return pd.DataFrame()


def read_data(path_to_data: str = ""):
    if path_to_data == "":
        path_to_data = os.listdir("data")
        if not path_to_data:
            return pd.DataFrame()
        path_to_data = os.path.join("data", path_to_data.pop())

    try:
        df = pd.read_json(f"{path_to_data}")
    except Exception:  # pylint: disable=broad-except
        print("Error while loading data")
        return pd.DataFrame()

    return df


def add_columns(df: pd.DataFrame, search_url: str) -> pd.DataFrame:
    df = df.copy()
    df["_source"] = search_url
    df["scrap_date"] = datetime.now()

    for column in LISTING_OPTIONAL_COLUMNS:
        if column not in df.columns:
            df[column] = None

    return df


def fill_nulls(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["price"] = df["price"].fillna("0")
    df["reviews_rating_number"] = df["reviews_rating_number"].fillna("0")
    df["reviews_amount"] = df["reviews_amount"].fillna("(0)")

    return df


def standardize_strings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["price"] = df["price"].astype(str).str.replace(".", "", regex=False)
    df["reviews_amount"] = df["reviews_amount"].astype(str).str.strip("()")

    return df


def price_to_float(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    try:
        df["price"] = df["price"].astype(float)
    except (ValueError, TypeError):
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df


def _ensure_product_details_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS product_details (
            item_id TEXT PRIMARY KEY,
            product_url TEXT,
            title_detail TEXT,
            price_detail TEXT,
            currency TEXT,
            condition TEXT,
            sold_quantity INTEGER,
            brand TEXT,
            model TEXT,
            color TEXT,
            material TEXT,
            capacity TEXT,
            voltage TEXT,
            seller_name TEXT,
            seller_location TEXT,
            official_store_flag INTEGER,
            shipping_full_flag INTEGER,
            shipping_free_flag INTEGER,
            rating_value REAL,
            rating_count INTEGER,
            description_plain TEXT,
            images_json TEXT,
            breadcrumbs_json TEXT,
            warranty_text TEXT,
            returns_text TEXT,
            scrap_date_detail TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_product_details_item_id
        ON product_details (item_id)
        """
    )


def _normalize_bool(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(bool(value))
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"1", "true", "yes", "on", "y", "si", "sí"}:
            return 1
        if value in {"0", "false", "no", "off", "n"}:
            return 0
    return int(bool(value))


def _normalize_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _normalize_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        normalized = str(value).replace(",", ".")
        return float(normalized)
    except (TypeError, ValueError):
        return None


def _normalize_json_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            json.loads(value)
        except json.JSONDecodeError:
            return json.dumps(value, ensure_ascii=False)
        return value
    if isinstance(value, (list, tuple, dict)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            return None
    return json.dumps(str(value), ensure_ascii=False)


def _prepare_detail_row(row: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for column in DETAIL_COLUMNS:
        value = row.get(column)
        if column in BOOL_COLUMNS:
            normalized[column] = _normalize_bool(value)
        elif column in INT_COLUMNS:
            normalized[column] = _normalize_int(value)
        elif column in FLOAT_COLUMNS:
            normalized[column] = _normalize_float(value)
        elif column in JSON_COLUMNS:
            normalized[column] = _normalize_json_value(value)
        elif column == "scrap_date_detail" and value:
            if isinstance(value, datetime):
                normalized[column] = value.isoformat()
            else:
                normalized[column] = str(value)
        else:
            normalized[column] = value
    return normalized


def upsert_product_details(conn: sqlite3.Connection, df_details: pd.DataFrame) -> None:
    if df_details.empty:
        return

    _ensure_product_details_table(conn)

    records = df_details.to_dict(orient="records")
    for record in records:
        payload = _prepare_detail_row(record)
        if not payload.get("item_id"):
            continue
        placeholders = ", ".join(f"{col} = excluded.{col}" for col in DETAIL_COLUMNS if col != "item_id")
        columns_clause = ", ".join(DETAIL_COLUMNS)
        values_placeholder = ", ".join(["?"] * len(DETAIL_COLUMNS))
        sql = (
            f"INSERT INTO product_details ({columns_clause}) "
            f"VALUES ({values_placeholder}) "
            f"ON CONFLICT(item_id) DO UPDATE SET {placeholders}"
        )
        conn.execute(sql, tuple(payload.get(col) for col in DETAIL_COLUMNS))


def save_to_sqlite3(
    df_listings: pd.DataFrame,
    details_df: pd.DataFrame | None = None,
    database_path: str | Path = "data/database.db",
) -> None:
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(database_path) as conn:
        df_listings.to_sql("mercadolivre_items", conn, if_exists="replace", index=False)
        if details_df is not None:
            upsert_product_details(conn, details_df)
        conn.commit()


def transform_data(
    path_to_data: str = "",
    search_url: str | None = None,
    details_path: str | Path | None = None,
):
    if path_to_data == "":
        path_to_data = "../data/data.json"

    df = read_data(path_to_data)
    if df.empty:
        print("No se encontraron datos para procesar.")
        return

    resolved_url = search_url or "https://listado.mercadolibre.com.ar/guitarra-electrica"
    df = add_columns(df, resolved_url)
    df = fill_nulls(df)
    df = standardize_strings(df)
    df = price_to_float(df)

    details_df = None
    if details_path is None:
        candidate = Path(path_to_data).parent / "product_details.json"
    else:
        candidate = Path(details_path)

    if candidate.exists():
        details_df = read_json_dataframe(candidate)
        if details_df.empty:
            details_df = None

    save_to_sqlite3(df, details_df)


if __name__ == "__main__":
    transform_data("../data/data.json")
