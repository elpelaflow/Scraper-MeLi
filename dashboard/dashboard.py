import sqlite3
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config_utils import load_search_query
from services.domain_discovery import (
    fetch_category_attributes,
    fetch_domain_discovery,
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


def get_dashboard(df: pd.DataFrame) -> None:
    st.sidebar.title("Pestaña de navegación")
    st.sidebar.markdown("Selecciona la fecha de scraping que deseas explorar.")

    with st.sidebar.expander("🧰 Herramientas", expanded=True):
        # Si ya tenés tabs para Básicos/Avanzados, mantenelos y solo agrega la de Domain Discovery.
        try:
            tab_basicos, tab_avanzados, tab_dd = st.tabs(["Básicos", "Avanzados", "Domain Discovery"])
        except Exception:
            # Fallback si no existen otras tabs
            tab_dd, = st.tabs(["Domain Discovery"])

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

    if df_to_show.empty:
        st.warning("No hay datos para los criterios seleccionados.")
    else:        st.dataframe(df_to_show, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Scraper Mercado Libre", layout="wide")

    data_dir = Path(__file__).resolve().parent.parent / "data" / "database.db"
    df = get_df_from_db(data_dir)

    if df.empty:
        st.sidebar.warning("No se encontró información en la base de datos.")

    get_dashboard(df)


if __name__ == "__main__":
    main()