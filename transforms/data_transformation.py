from __future__ import annotations

import json
import math
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable
from urllib.parse import urlparse

import pandas as pd
import numpy as np


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


def _ensure_listings_indexes(conn: sqlite3.Connection) -> None:
    try:
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_mercadolivre_items_source_scrap_date
            ON mercadolivre_items (_source, scrap_date)
            """
        )
    except sqlite3.OperationalError:
        pass

    try:
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_mercadolivre_items_item_id
            ON mercadolivre_items (item_id)
            """
        )
    except sqlite3.OperationalError:
        pass


def _ensure_pricing_kpis_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pricing_kpis (
            keyword_key TEXT NOT NULL,
            source_url TEXT NOT NULL,
            date TEXT NOT NULL,
            n INTEGER NOT NULL,
            min REAL,
            p25 REAL,
            p40 REAL,
            p50 REAL,
            p60 REAL,
            p75 REAL,
            max REAL,
            avg REAL,
            std REAL,
            computed_at TEXT NOT NULL,
            PRIMARY KEY (keyword_key, date)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_pricing_kpis_keyword_date
        ON pricing_kpis (keyword_key, date)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_pricing_kpis_date
        ON pricing_kpis (date)
        """
    )


def _extract_keyword_key(source_url: str | None) -> str:
    if not source_url:
        return "sin keyword"

    parsed = urlparse(str(source_url))
    path = (parsed.path or "").strip("/")
    if path:
        candidate = path.split("/")[-1]
    else:
        candidate = str(source_url).strip().split("/")[-1]

    keyword = candidate.replace("-", " ").strip()
    return keyword or "sin keyword"


def _safe_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def _safe_percentile(series: pd.Series, percentile: float) -> float | None:
    if series.empty:
        return None
    try:
        value = float(np.nanpercentile(series, percentile))
    except (ValueError, IndexError, TypeError):
        return None
    if math.isnan(value):
        return None
    return value


def _load_latest_batch(
    conn: sqlite3.Connection, source_url: str, limit: int = 50
) -> pd.DataFrame:
    latest_row = conn.execute(
        "SELECT MAX(scrap_date) FROM mercadolivre_items WHERE _source = ?",
        (source_url,),
    ).fetchone()

    if not latest_row or latest_row[0] is None:
        return pd.DataFrame()

    latest_scrap_date = latest_row[0]
    query = """
        SELECT
            mi.price,
            mi.scrap_date,
            mi.item_id
        FROM mercadolivre_items AS mi
        WHERE mi._source = ? AND mi.scrap_date = ?
        ORDER BY mi.rowid
        LIMIT ?
    """

    try:
        df = pd.read_sql_query(
            query,
            conn,
            params=(source_url, latest_scrap_date, limit),
        )
    except Exception:  # pylint: disable=broad-except
        return pd.DataFrame()

    return df


def _upsert_pricing_kpis_for_source(
    conn: sqlite3.Connection, source_url: str, limit: int = 50
) -> None:
    batch_df = _load_latest_batch(conn, source_url, limit=limit)
    if batch_df.empty or "price" not in batch_df.columns:
        return

    prices = pd.to_numeric(batch_df["price"], errors="coerce")
    prices = prices[prices > 0].dropna()
    if prices.empty:
        return

    scrap_dates = pd.to_datetime(batch_df["scrap_date"], errors="coerce")
    scrap_dates = scrap_dates.dropna()
    if scrap_dates.empty:
        target_date = datetime.now().date()
    else:
        target_date = scrap_dates.max().date()

    keyword_key = _extract_keyword_key(source_url)
    min_value = _safe_float(prices.min())
    max_value = _safe_float(prices.max())
    avg_value = _safe_float(prices.mean())
    if prices.size > 1:
        std_value = _safe_float(prices.std(ddof=0))
    else:
        std_value = 0.0

    payload_mapping = {
        "keyword_key": keyword_key,
        "source_url": source_url,
        "date": target_date.isoformat(),
        "n": int(prices.size),
        "min": min_value,
        "p25": _safe_percentile(prices, 25),
        "p40": _safe_percentile(prices, 40),
        "p50": _safe_percentile(prices, 50),
        "p60": _safe_percentile(prices, 60),
        "p75": _safe_percentile(prices, 75),
        "max": max_value,
        "avg": avg_value,
        "std": _safe_float(std_value),
        "computed_at": datetime.now().isoformat(timespec="seconds"),
    }

    columns = [
        "keyword_key",
        "source_url",
        "date",
        "n",
        "min",
        "p25",
        "p40",
        "p50",
        "p60",
        "p75",
        "max",
        "avg",
        "std",
        "computed_at",
    ]
    placeholders = ", ".join(["?"] * len(columns))
    update_columns = [
        "source_url",
        "n",
        "min",
        "p25",
        "p40",
        "p50",
        "p60",
        "p75",
        "max",
        "avg",
        "std",
        "computed_at",
    ]
    update_clause = ", ".join(
        f"{col} = excluded.{col}" for col in update_columns
    )

    values = tuple(payload_mapping[col] for col in columns)
    conn.execute(
        f"""
        INSERT INTO pricing_kpis ({', '.join(columns)})
        VALUES ({placeholders})
        ON CONFLICT(keyword_key, date) DO UPDATE SET {update_clause}
        """,
        values,
    )


def _update_pricing_kpis(conn: sqlite3.Connection, df_listings: pd.DataFrame) -> None:
    if df_listings.empty:
        return

    if "_source" not in df_listings.columns:
        return

    sources = [
        str(value)
        for value in df_listings["_source"].dropna().unique().tolist()
    ]
    if not sources:
        return

    for source_url in sources:
        try:
            _upsert_pricing_kpis_for_source(conn, source_url)
        except Exception:  # pylint: disable=broad-except
            continue


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
        df_listings.to_sql("mercadolivre_items", conn, if_exists="append", index=False)
        _ensure_listings_indexes(conn)
        if details_df is not None:
            upsert_product_details(conn, details_df)
        _ensure_pricing_kpis_table(conn)
        _update_pricing_kpis(conn, df_listings)
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
