import argparse
import datetime
import os

from extraction.spiders.mercadolivre import MercadoLivreSpider
from transforms.data_transformation import transform_data


def main(search_query: str):
    # Check if data.json exists
    data_path = os.path.abspath('data/data.json')

    if os.path.exists(data_path):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        new_name = f"data/data_{timestamp}.json"
        os.rename(data_path, new_name)
        print(f"Renamed existing data.json to data_{timestamp}.json")

    MercadoLivreSpider.run_spider(search_query)
    search_url = f"https://listado.mercadolibre.com.ar/{search_query.strip().lower().replace(' ', '-')}"
    transform_data('data/data.json', search_url)


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
    