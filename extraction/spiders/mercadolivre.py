from __future__ import annotations

import os
from typing import ClassVar

import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

save_path = os.path.join(os.getcwd(), 'data')


def _sanitize_query(search_query: str) -> str:
    """Convert the provided query into Mercado Libre URL format."""

    sanitized = search_query.strip().lower().replace(' ', '-')
    return sanitized or "guitarra-electrica"


class MercadoLivreSpider(scrapy.Spider):
    name = "mercadolivre"
    allowed_domains = ["listado.mercadolibre.com.ar"]

    default_query: ClassVar[str] = "guitarra-electrica"
    page_count = 1
    max_pages = 20

    def __init__(self, search_query: str | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.search_query = _sanitize_query(search_query or self.default_query)
        self.base_search_url = f"https://listado.mercadolibre.com.ar/{self.search_query}"
        self.start_urls = [self.base_search_url]
        self.page_count = 1


    def parse(self, response):
        products = response.css('div.ui-search-result__wrapper')

        for product in products:
            # Mercado Livre stores multiple "prices" in this single span
            prices = product.css('span.andes-money-amount__fraction::text').getall()

            yield {
                'name': product.css('a.poly-component__title::text').get(),
                'seller': product.css('span.poly-component__seller::text').get(),
                'price':prices[0] if len (prices) > 0 else None, 
                'reviews_rating_number': product.css('span.poly-reviews__rating::text').get(),
                'reviews_amount': product.css('span.poly-reviews__total::text').get()
            }

        if self.page_count < self.max_pages:
            # 48 is the amount of items shown in a given page
            offset = 48 * self.page_count
            next_page = self.base_search_url + f"{offset}_NoIndex_True"
            if next_page:
                self.page_count += 1
                yield scrapy.Request(url=next_page, callback=self.parse)


    def run_spider(search_query: str | None = None):
        os.makedirs(save_path, exist_ok=True)
        process = CrawlerProcess(settings={
            **get_project_settings(),
            "FEEDS": {
                os.path.join(save_path, 'data.json'): {
                    "format": "json",
                    "overwrite": True,
                }
            }
        })
        process.crawl(MercadoLivreSpider, search_query=search_query)
        process.start()
        