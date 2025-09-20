import sqlite3
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config_utils import load_search_query
from services.domain_discovery import fetch_domain_discovery


def get_df_from_db(db_path: str) -> pd.DataFrame:
  try:
    conn = sqlite3.connect(db_path)

    df = pd.read_sql_query("SELECT * FROM mercadolivre_items", conn)

  except:
    print("Error loading data")

  finally:
    conn.close()

  return df


def get_dashboard(df: pd.DataFrame):
    st.sidebar.title("Pestaña de navegación")
    st.sidebar.markdown("Selecciona la fecha de scraping que deseas explorar.")

    with st.sidebar.expander("🧰 Filtros", expanded=True):
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
            site = st.selectbox("Site", ["MLA"], index=0, help="Dejalo en MLA salvo que necesites otro")

            @st.cache_data(show_spinner=False, ttl=60*60)
            def _cached_domain_discovery(query: str, limit: int, site_code: str):
                return fetch_domain_discovery(query=query, limit=limit, site=site_code)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Refrescar"):
                    _cached_domain_discovery.clear()

            raw = _cached_domain_discovery(q, dd_limit, site)

            st.markdown("**Respuesta (JSON crudo):**")
            st.json(raw, expanded=False)

            # Vista rápida (si se puede)
            if isinstance(raw, list) and len(raw) > 0:
                import pandas as pd
                df_dd = pd.DataFrame(raw)
                keep = [c for c in ["domain_id", "domain_name", "category_id", "category_name", "relevance"] if c in df_dd.columns]
                if keep:
                    st.markdown("**Vista rápida (tabla):**")
                    st.dataframe(df_dd[keep], use_container_width=True, hide_index=True)

            # URL de ejemplo para copiar/pegar
            import urllib.parse
            url_example = "https://api.mercadolibre.com/sites/{}/domain_discovery/search?q={}&limit={}".format(
                site, urllib.parse.quote_plus(q or ""), dd_limit
            )
            st.code(f"GET {url_example}", language="bash")

            # (Opcional) Filtrar dataset principal por categoría sugerida si df existe en este scope:
            try:
                if isinstance(raw, list) and len(raw) > 0 and "category_id" in df.columns:
                    sugeridas = pd.DataFrame(raw)
                    if "category_id" in sugeridas.columns:
                        opciones = (
                            (sugeridas["category_name"].fillna("") + " (" + sugeridas["category_id"] + ")")
                            if "category_name" in sugeridas.columns
                            else sugeridas["category_id"]
                        ).dropna().drop_duplicates().tolist()
                        choice = st.selectbox("Filtrar dataset por categoría sugerida", ["(ninguna)"] + opciones)
                        if choice != "(ninguna)":
                            cat_elegida = choice.split("(")[-1].rstrip(")")
                            df = df[df["category_id"] == cat_elegida]
                elif isinstance(raw, list) and len(raw) > 0 and "category_id" not in (df.columns if 'df' in locals() else []):
                    st.warning("Tu dataset no incluye 'category_id'. Extraelo en el spider y guardalo para poder filtrar por categoría.")
            except Exception:
                pass

    scrap_dates = df.get("scrap_date")
    if scrap_dates is not None:
        available_dates = (
            scrap_dates.dropna().astype(str).unique().tolist()
        )
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
