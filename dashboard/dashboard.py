import json
import sqlite3
import unicodedata
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config_utils import load_bool_flag, load_search_query
from services.domain_discovery import (
    fetch_category_attributes,
    fetch_domain_discovery,
)


def _normalize_text(value: str) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn").lower()


def _format_currency(value: Any, currency: str | None = None) -> str:
    if value is None or value == "":
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    formatted = f"{value:,.0f}".replace(",", ".")
    return f"{formatted} {currency}".strip() if currency else formatted


def _format_badge(flag: Any) -> str:
    if flag is None:
        return "—"
    return "✅ Sí" if bool(flag) else "❌ No"


def _slugify(text: str) -> str:
    normalized = _normalize_text(text)
    slug = "-".join(segment for segment in normalized.split() if segment)
    return slug or "producto"


def _parse_json_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        if isinstance(parsed, list):
            return [str(item) for item in parsed if item]
        return [str(parsed)]
    return [str(value)]


def _build_txt_payload(detail_row: Dict[str, Any], listing_row: Dict[str, Any]) -> str:
    lines: List[str] = []

    title = detail_row.get("title_detail") or listing_row.get("name") or "Producto"
    item_id = detail_row.get("item_id") or listing_row.get("item_id") or "sin_id"
    lines.append(f"{title} ({item_id})")

    product_url = detail_row.get("product_url") or listing_row.get("product_url")
    if product_url:
        lines.extend(["", f"URL: {product_url}"])

    price_detail = detail_row.get("price_detail")
    currency = detail_row.get("currency")
    formatted_price = _format_currency(price_detail, currency)
    if formatted_price:
        lines.append(f"Precio: {formatted_price}")

    condition = detail_row.get("condition")
    if condition:
        lines.append(f"Condición: {condition}")

    sold_quantity = detail_row.get("sold_quantity")
    if sold_quantity is not None:
        lines.append(f"Vendidos: {sold_quantity}")

    attributes = []
    for key, label in (
        ("brand", "Marca"),
        ("model", "Modelo"),
        ("color", "Color"),
        ("material", "Material"),
        ("capacity", "Capacidad"),
        ("voltage", "Voltaje"),
    ):
        value = detail_row.get(key)
        if value:
            attributes.append(f"{label}: {value}")
    if attributes:
        lines.extend(["", "Atributos:"])
        lines.extend(attributes)

    seller_name = detail_row.get("seller_name")
    seller_location = detail_row.get("seller_location")
    if seller_name or seller_location:
        lines.extend(["", "Seller:"])
        if seller_name:
            lines.append(f"Nombre: {seller_name}")
        if seller_location:
            lines.append(f"Ubicación: {seller_location}")

    if detail_row.get("official_store_flag") is not None:
        lines.append(f"Tienda oficial: {'Sí' if detail_row.get('official_store_flag') else 'No'}")

    shipping_lines = []
    if detail_row.get("shipping_full_flag") is not None:
        shipping_lines.append(
            f"FULL: {'Sí' if detail_row.get('shipping_full_flag') else 'No'}"
        )
    if detail_row.get("shipping_free_flag") is not None:
        shipping_lines.append(
            f"Envío gratis: {'Sí' if detail_row.get('shipping_free_flag') else 'No'}"
        )
    if shipping_lines:
        lines.extend(["", "Envío:"])
        lines.extend(shipping_lines)

    rating_value = detail_row.get("rating_value")
    rating_count = detail_row.get("rating_count")
    if rating_value is not None or rating_count is not None:
        rating_text = "Reseñas:"
        if rating_value is not None:
            rating_text += f" promedio {rating_value}"
        if rating_count is not None:
            rating_text += f" ({rating_count} valoraciones)"
        lines.extend(["", rating_text])

    breadcrumbs = _parse_json_list(detail_row.get("breadcrumbs_json"))
    if breadcrumbs:
        lines.extend(["", "Categoría:"])
        lines.append(" > ".join(breadcrumbs))

    images = _parse_json_list(detail_row.get("images_json"))
    if images:
        lines.extend(["", "Imágenes:"])
        lines.extend(images)

    warranty = detail_row.get("warranty_text")
    if warranty:
        lines.extend(["", "Garantía:", warranty])

    returns = detail_row.get("returns_text")
    if returns:
        lines.extend(["", "Devoluciones:", returns])

    description = detail_row.get("description_plain")
    if description:
        lines.extend(["", "Descripción:", description])

    scrap_date = detail_row.get("scrap_date_detail")
    if scrap_date:
        lines.extend(["", f"Fecha de captura: {scrap_date}"])

    return "\n".join(lines).strip() + "\n"


def _format_listing_record(record: Dict[str, Any]) -> str:
    name = record.get("name") or "(sin nombre)"
    seller = record.get("seller")
    price = _format_currency(record.get("price"))
    parts = [name]
    extra: List[str] = []
    if seller:
        extra.append(str(seller))
    if price:
        extra.append(price)
    if extra:
        parts.append(" - " + " | ".join(extra))
    return "".join(parts)


def _load_details_table(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(db_path) as conn:
            return pd.read_sql_query("SELECT * FROM product_details", conn)
    except Exception:  # pylint: disable=broad-except
        return pd.DataFrame()


def _render_detail_preview(detail_row: Dict[str, Any], listing_row: Dict[str, Any]) -> None:
    product_title = detail_row.get("title_detail") or listing_row.get("name") or "—"
    product_url = detail_row.get("product_url") or listing_row.get("product_url")
    item_id = detail_row.get("item_id") or listing_row.get("item_id") or "—"

    top_col1, top_col2 = st.columns([3, 1])
    with top_col1:
        st.markdown("#### Producto")
        st.markdown(f"**Título:** {product_title}")
        st.markdown(f"**Item ID:** {item_id}")
        if product_url:
            st.markdown(f"[Ver ficha en Mercado Libre]({product_url})")
    with top_col2:
        scrap_date = detail_row.get("scrap_date_detail")
        st.markdown("#### Captura")
        st.markdown(f"Fecha: {scrap_date or '—'}")

    st.markdown("---")

    price_lines: List[str] = []
    formatted_price = _format_currency(detail_row.get("price_detail"), detail_row.get("currency"))
    if formatted_price:
        price_lines.append(f"**Precio:** {formatted_price}")
    condition = detail_row.get("condition")
    if condition:
        price_lines.append(f"**Condición:** {condition}")
    sold_quantity = detail_row.get("sold_quantity")
    if sold_quantity is not None:
        sold_formatted = f"{int(sold_quantity):,}".replace(",", ".")
        price_lines.append(f"**Vendidos:** {sold_formatted}")
    if price_lines:
        st.markdown("#### Precio")
        for line in price_lines:
            st.markdown(line)

    attribute_lines = []
    for key, label in (
        ("brand", "Marca"),
        ("model", "Modelo"),
        ("color", "Color"),
        ("material", "Material"),
        ("capacity", "Capacidad"),
        ("voltage", "Voltaje"),
    ):
        value = detail_row.get(key)
        if value:
            attribute_lines.append(f"**{label}:** {value}")
    if attribute_lines:
        st.markdown("#### Atributos")
        for line in attribute_lines:
            st.markdown(line)

    seller_lines = []
    seller_name = detail_row.get("seller_name")
    seller_location = detail_row.get("seller_location")
    if seller_name:
        seller_lines.append(f"**Seller:** {seller_name}")
    if seller_location:
        seller_lines.append(f"**Ubicación:** {seller_location}")
    official = detail_row.get("official_store_flag")
    if official is not None:
        seller_lines.append(f"**Tienda oficial:** {_format_badge(official)}")
    if seller_lines:
        st.markdown("#### Seller")
        for line in seller_lines:
            st.markdown(line)

    shipping_lines = []
    full_flag = detail_row.get("shipping_full_flag")
    free_flag = detail_row.get("shipping_free_flag")
    if full_flag is not None:
        shipping_lines.append(f"FULL: {_format_badge(full_flag)}")
    if free_flag is not None:
        shipping_lines.append(f"Envío gratis: {_format_badge(free_flag)}")
    if shipping_lines:
        st.markdown("#### Envío")
        for line in shipping_lines:
            st.markdown(line)

    rating_value = detail_row.get("rating_value")
    rating_count = detail_row.get("rating_count")
    if rating_value is not None or rating_count is not None:
        st.markdown("#### Reseñas")
        rating_text = []
        if rating_value is not None:
            rating_text.append(f"Promedio: {rating_value}")
        if rating_count is not None:
            rating_text.append(f"Cantidad: {rating_count}")
        st.markdown(" | ".join(rating_text))

    breadcrumbs = _parse_json_list(detail_row.get("breadcrumbs_json"))
    if breadcrumbs:
        st.markdown("#### Categoría")
        st.markdown(" > ".join(breadcrumbs))

    images = _parse_json_list(detail_row.get("images_json"))[:3]
    if images:
        st.markdown("#### Imágenes")
        st.image(images, width=160, caption=None)

    description = detail_row.get("description_plain")
    if description:
        st.markdown("#### Descripción")
        lines = description.splitlines()
        preview = "\n".join(lines[:10])
        st.text(preview)
        if len(lines) > 10:
            with st.expander("Ver descripción completa"):
                st.text(description)

    warranty = detail_row.get("warranty_text")
    returns = detail_row.get("returns_text")
    if warranty or returns:
        st.markdown("#### Garantía y devoluciones")
        if warranty:
            st.markdown(f"**Garantía:** {warranty}")
        if returns:
            st.markdown(f"**Devoluciones:** {returns}")


def _render_info_product_tab(tab_container, df: pd.DataFrame) -> None:
    db_path = Path(__file__).resolve().parent.parent / "data" / "database.db"

    with tab_container:
        st.markdown("### Info product")
        flag_enabled = load_bool_flag("ENABLE_PRODUCT_DETAILS", True)
        st.caption(
            "Captura de detalles {}. Cambiá la variable de entorno "
            "ENABLE_PRODUCT_DETAILS para activar o desactivar el seguimiento.".format(
                "habilitada" if flag_enabled else "deshabilitada"
            )
        )

        if df.empty:
            st.info(
                "No hay datos del listado para buscar productos. Ejecutá el scraping "
                "para generar resultados."
            )
            return

        working_df = df.copy()
        for column in ["name", "seller", "price", "item_id", "product_url"]:
            if column not in working_df.columns:
                working_df[column] = None

        @st.cache_data(show_spinner=False)
        def _cached_index(dataframe: pd.DataFrame) -> List[Dict[str, Any]]:
            records: List[Dict[str, Any]] = []
            for _, row in dataframe.iterrows():
                name = (row.get("name") or "").strip()
                if not name:
                    continue
                record: Dict[str, Any] = {
                    "name": name,
                    "normalized_name": _normalize_text(name),
                    "seller": row.get("seller"),
                    "price": row.get("price"),
                    "item_id": row.get("item_id"),
                    "product_url": row.get("product_url"),
                }
                record["key"] = f"{record['item_id'] or ''}|{record['product_url'] or ''}|{record['name']}"
                records.append(record)
            return records

        index_records = _cached_index(working_df[["name", "seller", "price", "item_id", "product_url"]])

        if not index_records:
            st.info("No hay nombres de productos disponibles para la búsqueda.")
            return

        search_query = st.text_input("Buscar producto por nombre", key="info_product_search")
        normalized_query = _normalize_text(search_query)

        selected_record: Dict[str, Any] | None = None
        suggestions: List[Dict[str, Any]] = []

        if normalized_query:
            for record in index_records:
                if normalized_query in record["normalized_name"]:
                    suggestions.append(record)
                if len(suggestions) >= 10:
                    break

            if suggestions:
                option_indices = list(range(len(suggestions)))
                previous_key = st.session_state.get("info_product_selected_key")
                default_index = 0
                if previous_key is not None:
                    for idx, rec in enumerate(suggestions):
                        if rec["key"] == previous_key:
                            default_index = idx
                            break

                selected_idx = st.selectbox(
                    "Coincidencias",
                    options=option_indices,
                    format_func=lambda idx: _format_listing_record(suggestions[idx]),
                    index=default_index if option_indices else 0,
                    key="info_product_selection",
                )
                if option_indices:
                    selected_record = suggestions[selected_idx]
                    st.session_state["info_product_selected_key"] = selected_record["key"]
            else:
                exact_matches = [rec for rec in index_records if rec["normalized_name"] == normalized_query]
                if len(exact_matches) == 1:
                    selected_record = exact_matches[0]
                    st.session_state["info_product_selected_key"] = selected_record["key"]
                else:
                    st.info("No se encontraron coincidencias para el término ingresado.")
        else:
            st.info("Escribí parte del nombre para ver sugerencias en vivo.")

        if not selected_record:
            return

        @st.cache_data(show_spinner=False)
        def _cached_details(path_str: str) -> pd.DataFrame:
            return _load_details_table(Path(path_str))

        details_df = _cached_details(str(db_path))

        detail_payload: Dict[str, Any] | None = None
        if not details_df.empty:
            if "item_id" in details_df.columns and selected_record.get("item_id"):
                matches = details_df[details_df["item_id"] == selected_record["item_id"]]
                if not matches.empty:
                    detail_payload = matches.iloc[0].to_dict()
            if detail_payload is None and selected_record.get("product_url") and "product_url" in details_df.columns:
                matches = details_df[details_df["product_url"] == selected_record["product_url"]]
                if not matches.empty:
                    detail_payload = matches.iloc[0].to_dict()

        listing_payload = dict(selected_record)

        if detail_payload is None:
            st.warning(
                "Este producto todavía no tiene detalles capturados. Ejecutá el scraping "
                "con detalles habilitados (flag ENABLE_PRODUCT_DETAILS=true) y volvé a intentarlo."
            )
            product_url = listing_payload.get("product_url")
            if product_url:
                st.markdown(f"[Abrir producto en Mercado Libre]({product_url})")
            return

        _render_detail_preview(detail_payload, listing_payload)

        txt_payload = _build_txt_payload(detail_payload, listing_payload)
        date_value = detail_payload.get("scrap_date_detail")
        if isinstance(date_value, str) and date_value:
            date_part = date_value.split("T")[0].replace("-", "")
        else:
            date_part = datetime.now().strftime("%Y%m%d")
        title_for_slug = detail_payload.get("title_detail") or listing_payload.get("name") or "producto"
        file_name = (
            f"{detail_payload.get('item_id') or listing_payload.get('item_id') or 'producto'}"
            f"_{_slugify(title_for_slug)}_{date_part}.txt"
        )
        st.download_button(
            "Descargar .txt",
            data=txt_payload.encode("utf-8"),
            file_name=file_name,
            mime="text/plain",
        )


def get_df_from_db(db_path: str | Path) -> pd.DataFrame:
    """Load items from the SQLite database.

    Returns an empty dataframe if the database is not available or the
    query fails so the Streamlit app can continue rendering.
    """

    db_path = Path(db_path)

    if not db_path.exists():
        return pd.DataFrame()

    try:
        with sqlite3.connect(db_path) as conn:
            return pd.read_sql_query("SELECT * FROM mercadolivre_items", conn)
    except Exception as exc:  # pylint: disable=broad-except
        # Streamlit logs printed exceptions to the terminal, which is
        # enough feedback here. Returning an empty dataframe keeps the UI
        # responsive instead of failing with a blank page.
        print(f"Error loading data from {db_path}: {exc}")
        return pd.DataFrame()


def load_all_tables(db_path: str | Path) -> Dict[str, pd.DataFrame]:
    """Return all tables stored in the SQLite database as dataframes."""

    db_path = Path(db_path)

    if not db_path.exists():
        return {}

    tables: Dict[str, pd.DataFrame] = {}

    try:
        with sqlite3.connect(db_path) as conn:
            try:
                result = pd.read_sql_query(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
                    conn,
                )
                table_names = result["name"].tolist()
            except Exception as exc:  # pylint: disable=broad-except
                print(f"Error retrieving table list from {db_path}: {exc}")
                table_names = []

            for table_name in table_names:
                try:
                    tables[table_name] = pd.read_sql_query(
                        f'SELECT * FROM "{table_name}"', conn
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    print(f"Error loading table {table_name} from {db_path}: {exc}")
                    tables[table_name] = pd.DataFrame()
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error opening database {db_path}: {exc}")

    return tables


def get_dashboard(
    df: pd.DataFrame, tables: Dict[str, pd.DataFrame] | None = None
) -> None:
    tables = tables or {}
    st.sidebar.title("Pestaña de navegación")
    st.sidebar.markdown("Selecciona la fecha de scraping que deseas explorar.")

    with st.sidebar.expander("🧰 Herramientas", expanded=True):
        # Si ya tenés tabs para Básicos/Avanzados, mantenelos y solo agrega la de Domain Discovery.
        try:
            tab_basicos, tab_avanzados, tab_info_product, tab_dd = st.tabs(
                ["Básicos", "Avanzados", "Info product", "Domain Discovery"]
            )
        except Exception:
            # Fallback si no existen otras tabs
            tab_info_product, tab_dd = st.tabs(["Info product", "Domain Discovery"])

        _render_info_product_tab(tab_info_product, df)

        with tab_dd:
            q = load_search_query()  # misma búsqueda que definiste en search_ui
            st.caption(f"Búsqueda actual: **{q}**")

            dd_limit = st.slider("Límite del llamado", 1, 20, 5, key="dd_limit")
            site = st.selectbox(
                "Site", ["MLA"], index=0, help="Dejalo en MLA salvo que necesites otro"
            )

            @st.cache_data(show_spinner=False, ttl=60 * 60)
            def _cached_domain_discovery(query: str, limit: int, site_code: str):
                return fetch_domain_discovery(query=query, limit=limit, site=site_code)

            @st.cache_data(show_spinner=False, ttl=60 * 60)
            def _cached_category_attrs(category_id: str, site_code: str):
                return fetch_category_attributes(category_id=category_id, site=site_code)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Refrescar Domain Discovery"):
                    _cached_domain_discovery.clear()
            with col2:
                if st.button("Limpiar caché de atributos"):
                    _cached_category_attrs.clear()

            results, raw_body = _cached_domain_discovery(q, dd_limit, site)

            tab_overview, tab_attrs = st.tabs([
                "Visión general",
                "Atributos de categorías",
            ])

            with tab_overview:
                st.markdown("**Respuesta (JSON crudo):**")
                if raw_body:
                    st.code(raw_body, language="json")
                else:
                    st.info("No se obtuvo respuesta del servicio de Domain Discovery.")

                if results:
                    st.markdown("**Vista estructurada:**")
                    st.json(results, expanded=False)

                # Vista rápida (si se puede)
                if isinstance(results, list) and len(results) > 0:
                    df_dd = pd.DataFrame(results)
                    keep = [
                        c
                        for c in [
                            "domain_id",
                            "domain_name",
                            "category_id",
                            "category_name",
                            "relevance",
                        ]
                        if c in df_dd.columns
                    ]
                    if keep:
                        st.markdown("**Vista rápida (tabla):**")
                        st.dataframe(
                            df_dd[keep], use_container_width=True, hide_index=True
                        )

                # URL de ejemplo para copiar/pegar
                import urllib.parse

                url_example = "https://api.mercadolibre.com/sites/{}/domain_discovery/search?q={}&limit={}".format(
                    site, urllib.parse.quote_plus(q or ""), dd_limit
                )
                st.code(f"GET {url_example}", language="bash")

                # (Opcional) Filtrar dataset principal por categoría sugerida si df existe en este scope:
                try:
                    if (
                        isinstance(results, list)
                        and len(results) > 0
                        and "category_id" in df.columns
                    ):
                        sugeridas = pd.DataFrame(results)
                        if "category_id" in sugeridas.columns:
                            opciones = (
                                (
                                    sugeridas["category_name"].fillna("")
                                    + " ("
                                    + sugeridas["category_id"]
                                    + ")"
                                )
                                if "category_name" in sugeridas.columns
                                else sugeridas["category_id"]
                            ).dropna().drop_duplicates().tolist()
                            choice = st.selectbox(
                                "Filtrar dataset por categoría sugerida",
                                ["(ninguna)"] + opciones,
                            )
                            if choice != "(ninguna)":
                                cat_elegida = choice.split("(")[-1].rstrip(")")
                                df = df[df["category_id"] == cat_elegida]
                    elif isinstance(results, list) and len(results) > 0 and "category_id" not in (
                        df.columns if "df" in locals() else []
                    ):
                        st.warning(
                            "Tu dataset no incluye 'category_id'. Extraelo en el spider y guardalo para poder filtrar por categoría."
                        )
                except Exception:
                    pass

            with tab_attrs:
                if isinstance(results, list):
                    category_mapping: dict[str, str] = {}
                    for item in results:
                        if not isinstance(item, dict):
                            continue
                        cat_id = item.get("category_id")
                        if not cat_id:
                            continue
                        cat_id = str(cat_id)
                        cat_name = item.get("category_name")
                        if cat_name:
                            category_mapping[cat_id] = str(cat_name)
                        else:
                            category_mapping.setdefault(cat_id, "")
                    category_ids = sorted(cid for cid in category_mapping.keys() if cid)
                else:
                    category_ids = []
                    category_mapping = {}

                if not category_ids:
                    st.info(
                        "No se detectaron categorías en los resultados. Ejecutá Domain Discovery para habilitar esta sección."
                    )
                else:
                    for cat_id in category_ids:
                        cat_name = category_mapping.get(cat_id, "")
                        display_name = cat_name or "Categoría sin nombre"
                        with st.spinner(
                            f"Consultando atributos para {display_name} ({cat_id})..."
                        ):
                            attrs, raw_attrs = _cached_category_attrs(cat_id, site)

                        st.markdown(f"#### {display_name} ({cat_id})")

                        if not raw_attrs:
                            st.warning(
                                "No se pudieron obtener los atributos de la categoría. Intentalo nuevamente más tarde."
                            )
                            st.divider()
                            continue

                        st.markdown("**Respuesta (JSON crudo):**")
                        st.code(raw_attrs, language="json")

                        if attrs:
                            st.markdown("**Vista estructurada:**")
                            st.json(attrs, expanded=False)
                            try:
                                df_attrs = pd.json_normalize(attrs)
                            except Exception:
                                df_attrs = pd.DataFrame()
                            if not df_attrs.empty:
                                keep_columns = [
                                    col
                                    for col in [
                                        "id",
                                        "name",
                                        "value_type",
                                        "tags.required",
                                        "tags.restricted_values",
                                        "tags.inference_priority",
                                    ]
                                    if col in df_attrs.columns
                                ]
                                if keep_columns:
                                    st.markdown("**Vista rápida (tabla):**")
                                    st.dataframe(
                                        df_attrs[keep_columns],
                                        use_container_width=True,
                                        hide_index=True,
                                    )
                        else:
                            st.info("La categoría no devolvió atributos para mostrar.")

                        st.divider()

    scrap_dates = df.get("scrap_date")
    if scrap_dates is not None:
        available_dates = scrap_dates.dropna().astype(str).unique().tolist()
    else:
        available_dates = []

    available_dates.sort(reverse=True)
    selected_date = st.sidebar.selectbox(
        "Fecha de scraping",
        options=["Todas las fechas"] + available_dates,
        index=0,
    )

    if selected_date != "Todas las fechas":
        df_to_show = df[df["scrap_date"].astype(str) == selected_date]
    else:
        df_to_show = df

    st.title("Resultados del scraping de Mercado Libre Argentina")
    st.subheader("By Heitor Nolla")

    tab_labels = ["Resultados"]
    if tables:
        tab_labels.append("Tablas disponibles")

    tabs = st.tabs(tab_labels)

    with tabs[0]:
        if df_to_show.empty:
            st.warning("No hay datos para los criterios seleccionados.")
        else:
            st.dataframe(df_to_show, use_container_width=True)

        if not tables:
            st.info(
                "No se detectaron tablas adicionales en la base de datos o no fue posible cargarlas."
            )

    if tables:
        with tabs[1]:
            table_names = sorted(tables.keys())
            selected_table = st.selectbox(
                "Seleccioná la tabla que querés explorar",
                options=table_names,
                index=0,
                key="table_selector",
            )

            selected_df = tables.get(selected_table, pd.DataFrame())
            if selected_df.empty:
                st.info(
                    "La tabla seleccionada no tiene registros para mostrar en este momento."
                )
            else:
                st.dataframe(selected_df, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Scraper Mercado Libre", layout="wide")

    data_dir = Path(__file__).resolve().parent.parent / "data" / "database.db"
    df = get_df_from_db(data_dir)
    tables = load_all_tables(data_dir)

    if df.empty:
        st.sidebar.warning("No se encontró información en la base de datos.")

    get_dashboard(df, tables)


if __name__ == "__main__":
    main()