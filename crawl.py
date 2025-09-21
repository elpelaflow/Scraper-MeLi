import argparse
import datetime
import os
from pathlib import Path

from extraction.spiders.mercadolivre import MercadoLivreSpider
from transforms.data_transformation import transform_data


def main(search_query: str):
    # Check if data.json exists
    data_dir = Path('data')
    data_dir.mkdir(parents=True, exist_ok=True)

    data_path = data_dir / 'data.json'
    details_path = data_dir / 'product_details.json'

    if data_path.exists():
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        new_name = data_dir / f"data_{timestamp}.json"
        data_path.rename(new_name)
        print(f"Renamed existing data.json to {new_name.name}")

    if details_path.exists():
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        new_name = data_dir / f"product_details_{timestamp}.json"
        details_path.rename(new_name)
        print(f"Renamed existing product_details.json to {new_name.name}")

    MercadoLivreSpider.run_spider(search_query)
    search_url = f"https://listado.mercadolibre.com.ar/{search_query.strip().lower().replace(' ', '-')}"
    transform_data(str(data_path), search_url, details_path=str(details_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Mercado Libre crawler")
    parser.add_argument(
        "--query",
        "-q",
        default="guitarra electrica",
        help="Search term to use when scraping Mercado Libre.",
    )
    args = parser.parse_args()
    main(args.query)
    