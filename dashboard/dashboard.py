+4
-4

import pandas as pd
import sqlite3
import streamlit as st


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
    st.title("Resultados del scraping de Mercado Libre Argentina")
    st.subheader("By Heitor Nolla")

    col1, col2 = st.columns(2)

    total_itens = df.shape[0]
    col1.metric(label="Total Items Found", value=total_itens)

    avg_price = df['price'].mean()
    col2.metric(label="Precio promedio (ARS)", value=f"{avg_price:.2f}")

    st.markdown("### Todos los artículos")

    # Search input
    search_term = st.text_input("Buscar productos (por palabra clave):")

    # Filter dataframe based on search input
    if search_term:
        filtered_df = df[df.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)]
    else:
        filtered_df = df

    st.dataframe(filtered_df.reset_index(drop=True))



if __name__ == "__main__":
  df = get_df_from_db('data/database.db')
  get_dashboard(df)
  