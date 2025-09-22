from __future__ import annotations

import json
import os
import re
from datetime import datetime
from html import unescape
from typing import ClassVar, Iterable

import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.http import Response
from scrapy.utils.project import get_project_settings

from config_utils import load_bool_flag, load_int_flag, resolve_max_pages
from extraction.items import ListingItem, ProductDetailItem

save_path = os.path.join(os.getcwd(), "data")


def _sanitize_query(search_query: str) -> str:
    """Convert the provided query into Mercado Libre URL format."""

    sanitized = search_query.strip().lower().replace(" ", "-")
    return sanitized or "guitarra-electrica"


class MercadoLivreSpider(scrapy.Spider):
    name = "mercadolivre"
    allowed_domains = [
        "listado.mercadolibre.com.ar",
        "www.mercadolibre.com.ar",
        "articulo.mercadolibre.com.ar",
    ]

    default_query: ClassVar[str] = "guitarra-electrica"
    def __init__(self, search_query: str | None = None, *args, **kwargs):
        self.enable_product_details = load_bool_flag("ENABLE_PRODUCT_DETAILS", True)
        details_limit = load_int_flag("DETAILS_MAX_PER_RUN", None)

        try:
            override_flag = kwargs.pop("enable_product_details")
        except KeyError:
            override_flag = None
        if override_flag is not None:
            self.enable_product_details = bool(override_flag)

        try:
            override_limit = kwargs.pop("details_max_per_run")
        except KeyError:
            override_limit = None
        if override_limit is not None:
            try:
                details_limit = int(override_limit)
            except (TypeError, ValueError):
                details_limit = details_limit

        try:
            requested_max_pages = kwargs.pop("max_pages")
        except KeyError:
            requested_max_pages = None

        super().__init__(*args, **kwargs)

        self.detail_limit = details_limit if details_limit and details_limit > 0 else None
        self.detail_counter = 0
        self.visited_detail_ids: set[str] = set()

        self.search_query = _sanitize_query(search_query or self.default_query)
        self.base_search_url = f"https://listado.mercadolibre.com.ar/{self.search_query}"
        self.start_urls = [self.base_search_url]
        self.max_pages_requested = resolve_max_pages(cli_value=requested_max_pages)
        self.max_pages = self.max_pages_requested
        self.pages_fetched = 0

    def start_requests(self):  # type: ignore[override]
        self.logger.info(
            "Iniciando scraping para '%s' (max_pages_requested=%s)",
            self.search_query,
            self.max_pages_requested,
        )
        yield from super().start_requests()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_item_id(product_url: str | None) -> str | None:
        if not product_url:
            return None
        match = re.search(r"/(ML[A-Z]{1,3}-?\d+)", product_url, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        return None

    @staticmethod
    def _strip_text(iterable: Iterable[str] | None) -> list[str]:
        if not iterable:
            return []
        stripped: list[str] = []
        for value in iterable:
            value = unescape(value or "").strip()
            if value:
                stripped.append(value)
        return stripped

    @staticmethod
    def _combine_price(parts: Iterable[str]) -> str | None:
        tokens = [re.sub(r"\D", "", chunk) for chunk in parts if chunk]
        combined = "".join(tokens)
        return combined or None

    @staticmethod
    def _to_int(value: str | None) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_json_loads(payload: str | None) -> dict | list | None:
        if not payload:
            return None
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
    def parse(self, response: Response):
        products = response.css("div.ui-search-result__wrapper")
        product_count = len(products)
        self.pages_fetched += 1

        for product in products:
            prices = product.css("span.andes-money-amount__fraction::text").getall()
            product_url = product.css("a.poly-component__title::attr(href)").get()
            if product_url:
                product_url = response.urljoin(product_url)
            item_id = self._extract_item_id(product_url)

            listing_item = ListingItem(
                name=product.css("a.poly-component__title::text").get(),
                seller=product.css("span.poly-component__seller::text").get(),
                price=prices[0] if len(prices) > 0 else None,
                reviews_rating_number=product.css("span.poly-reviews__rating::text").get(),
                reviews_amount=product.css("span.poly-reviews__total::text").get(),
                product_url=product_url,
                item_id=item_id,
            )

            yield listing_item

            if self._should_visit_detail(item_id, product_url):
                yield response.follow(
                    product_url,
                    callback=self.parse_product_detail,
                    cb_kwargs={"item_id": item_id, "product_url": product_url},
                    meta={"referer": response.url},
                )

        if product_count == 0:
            self.logger.info(
                "Página %s sin resultados. Deteniendo paginación.",
                self.pages_fetched,
            )
            return

        if self.pages_fetched >= self.max_pages_requested:
            self.logger.info(
                "Se alcanzó el máximo de páginas solicitadas (%s).",
                self.max_pages_requested,
            )
            return

        if product_count < 48:
            self.logger.info(
                "Página %s con %s resultados (<48). Deteniendo paginación.",
                self.pages_fetched,
                product_count,
            )
            return

        offset = 48 * self.pages_fetched
        next_page = self.base_search_url + f"{offset}_NoIndex_True"
        self.logger.debug(
            "Solicitando página %s (offset=%s, max_pages=%s)",
            self.pages_fetched + 1,
            offset,
            self.max_pages_requested,
        )
        yield scrapy.Request(url=next_page, callback=self.parse)

    def _should_visit_detail(self, item_id: str | None, product_url: str | None) -> bool:
        if not self.enable_product_details:
            return False
        if not item_id or not product_url:
            return False
        if item_id in self.visited_detail_ids:
            return False
        if self.detail_limit is not None and self.detail_counter >= self.detail_limit:
            return False
        self.visited_detail_ids.add(item_id)
        self.detail_counter += 1
        return True

    # pylint: disable=too-many-locals
    def parse_product_detail(self, response: Response, item_id: str, product_url: str):
        product_ld: dict | None = None
        breadcrumbs_ld: dict | None = None

        for script in response.xpath("//script[@type='application/ld+json']/text()").getall():
            parsed = self._safe_json_loads(script)
            if isinstance(parsed, list):
                for entry in parsed:
                    if isinstance(entry, dict):
                        product_ld, breadcrumbs_ld = self._assign_ld_json(entry, product_ld, breadcrumbs_ld)
            elif isinstance(parsed, dict):
                product_ld, breadcrumbs_ld = self._assign_ld_json(parsed, product_ld, breadcrumbs_ld)

        title = self._get_title(response, product_ld)
        price_detail, currency = self._get_price_info(response, product_ld)
        condition, sold_quantity = self._get_condition_and_sales(response)
        rating_value, rating_count = self._get_rating(product_ld, response)
        description_plain = self._get_description(product_ld, response)
        images = self._get_images(product_ld, response)
        breadcrumbs = self._get_breadcrumbs(breadcrumbs_ld, response)
        seller_name, seller_location, official_store_flag = self._get_seller_info(response)
        shipping_full_flag, shipping_free_flag = self._get_shipping_flags(response)
        warranty_text, returns_text = self._get_warranty_and_returns(response)
        attributes = self._get_attributes(response)

        detail_item = ProductDetailItem(
            item_id=item_id,
            product_url=product_url,
            title_detail=title,
            price_detail=price_detail,
            currency=currency,
            condition=condition,
            sold_quantity=sold_quantity,
            brand=attributes.get("marca"),
            model=attributes.get("modelo"),
            color=attributes.get("color"),
            material=attributes.get("material"),
            capacity=attributes.get("capacidad"),
            voltage=attributes.get("voltaje"),
            seller_name=seller_name,
            seller_location=seller_location,
            official_store_flag=official_store_flag,
            shipping_full_flag=shipping_full_flag,
            shipping_free_flag=shipping_free_flag,
            rating_value=rating_value,
            rating_count=rating_count,
            description_plain=description_plain,
            images_json=images,
            breadcrumbs_json=breadcrumbs,
            warranty_text=warranty_text,
            returns_text=returns_text,
            scrap_date_detail=datetime.utcnow().isoformat(),
        )

        yield detail_item

    # ------------------------------------------------------------------
    # Detail helpers
    # ------------------------------------------------------------------
    def _assign_ld_json(
        self,
        entry: dict,
        product_ld: dict | None,
        breadcrumbs_ld: dict | None,
    ) -> tuple[dict | None, dict | None]:
        entry_type = entry.get("@type")
        if entry_type == "Product" and product_ld is None:
            product_ld = entry
        elif entry_type == "BreadcrumbList" and breadcrumbs_ld is None:
            breadcrumbs_ld = entry
        return product_ld, breadcrumbs_ld

    def _get_title(self, response: Response, product_ld: dict | None) -> str | None:
        title = None
        if product_ld:
            title = product_ld.get("name")
        if not title:
            title = response.css("h1::text").get()
        return title.strip() if isinstance(title, str) else title

    def _get_price_info(self, response: Response, product_ld: dict | None) -> tuple[str | None, str | None]:
        if product_ld:
            offers = product_ld.get("offers")
            if isinstance(offers, dict):
                price = offers.get("price")
                currency = offers.get("priceCurrency")
                if price is not None:
                    return str(price), currency

        price_chunks = response.css("span.andes-money-amount__fraction::text").getall()
        cents = response.css("span.andes-money-amount__cents::text").get()
        if cents:
            price_chunks.append(cents)
        price = self._combine_price(price_chunks)
        currency = response.css("span.andes-money-amount__currency-symbol::text").get()
        return price, currency

    def _get_condition_and_sales(self, response: Response) -> tuple[str | None, int | None]:
        condition = None
        sold_quantity = None

        subtitle_texts = self._strip_text(response.css("span.ui-pdp-subtitle::text").getall())
        for text in subtitle_texts:
            parts = [segment.strip() for segment in text.split("|") if segment.strip()]
            for part in parts:
                lowered = part.lower()
                if "vendid" in lowered:
                    sold_quantity = self._to_int(re.sub(r"\D", "", lowered))
                elif not condition:
                    condition = part

        if sold_quantity is None:
            sold_text = response.xpath("//*[contains(translate(text(),'VENDIDOS','vendidos'),'vendidos')]/text()").get()
            if sold_text:
                sold_quantity = self._to_int(re.sub(r"\D", "", sold_text))

        return condition, sold_quantity

    def _get_rating(self, product_ld: dict | None, response: Response) -> tuple[float | None, int | None]:
        rating_value = None
        rating_count = None

        if product_ld:
            rating = product_ld.get("aggregateRating")
            if isinstance(rating, dict):
                rating_value = rating.get("ratingValue")
                rating_count = rating.get("ratingCount") or rating.get("reviewCount")

        if rating_value is None:
            rating_value_text = response.css("span.ui-pdp-review__rating::text").get()
            if rating_value_text:
                try:
                    rating_value = float(rating_value_text.replace(",", "."))
                except ValueError:
                    rating_value = None

        if rating_count is None:
            count_text = response.css("span.ui-pdp-review__amount::text").re_first(r"\d+")
            rating_count = self._to_int(count_text)

        if isinstance(rating_value, str):
            try:
                rating_value = float(rating_value.replace(",", "."))
            except ValueError:
                rating_value = None

        if isinstance(rating_count, str):
            rating_count = self._to_int(rating_count)

        return rating_value, rating_count

    def _get_description(self, product_ld: dict | None, response: Response) -> str | None:
        if product_ld:
            description = product_ld.get("description")
            if isinstance(description, str) and description.strip():
                return unescape(description.strip())

        paragraphs = response.css("div.ui-pdp-description__content *::text").getall()
        description_text = "\n".join(self._strip_text(paragraphs))
        return description_text or None

    def _get_images(self, product_ld: dict | None, response: Response) -> list[str]:
        urls: list[str] = []
        if product_ld:
            images = product_ld.get("image")
            if isinstance(images, str):
                urls.append(images)
            elif isinstance(images, list):
                urls.extend(str(img) for img in images if img)

        gallery = response.css("figure.ui-pdp-gallery__figure img::attr(data-zoom)").getall()
        if not gallery:
            gallery = response.css("figure.ui-pdp-gallery__figure img::attr(src)").getall()
        urls.extend(self._strip_text(gallery))

        seen: set[str] = set()
        unique_urls: list[str] = []
        for url in urls:
            if not url:
                continue
            if url in seen:
                continue
            seen.add(url)
            unique_urls.append(url)
        return unique_urls

    def _get_breadcrumbs(self, breadcrumbs_ld: dict | None, response: Response) -> list[str]:
        breadcrumbs: list[str] = []
        if breadcrumbs_ld and isinstance(breadcrumbs_ld.get("itemListElement"), list):
            for node in breadcrumbs_ld["itemListElement"]:
                if not isinstance(node, dict):
                    continue
                item = node.get("item")
                if isinstance(item, dict):
                    name = item.get("name")
                else:
                    name = node.get("name")
                if isinstance(name, str) and name.strip():
                    breadcrumbs.append(name.strip())

        if not breadcrumbs:
            breadcrumbs = self._strip_text(
                response.css("nav.ui-pdp-breadcrumb__container a::text").getall()
            )
        return breadcrumbs

    def _get_seller_info(self, response: Response) -> tuple[str | None, str | None, bool]:
        seller_name = response.css("div.ui-pdp-seller__profile-name::text").get()
        if not seller_name:
            seller_name = response.css("span.ui-pdp-seller__header__title::text").get()
        seller_name = seller_name.strip() if seller_name else None

        location = response.css("div.ui-pdp-seller__header__subtitle::text").get()
        if not location:
            location = " ".join(
                self._strip_text(response.css("div.ui-seller-info__status-value::text").getall())
            )
        location = location.strip() if isinstance(location, str) else location

        official_flag = bool(
            response.xpath(
                "//*[contains(translate(text(),'TIENDA OFICIAL','tienda oficial'),'tienda oficial')]"
            ).get()
        )

        return seller_name, location, official_flag

    def _get_shipping_flags(self, response: Response) -> tuple[bool, bool]:
        full_flag = bool(
            response.xpath(
                "//*[contains(@class, 'full') or contains(translate(text(),'FULL','full'),'full')][contains(text(),'FULL') or contains(translate(text(),'FULL','full'),'full')]"
            ).get()
        )
        free_flag = bool(
            response.xpath(
                "//*[contains(translate(text(),'ENVÍO GRATIS','envío gratis'),'envío gratis') or contains(translate(text(),'ENVIO GRATIS','envio gratis'),'envio gratis')]"
            ).get()
        )
        return full_flag, free_flag

    def _get_warranty_and_returns(self, response: Response) -> tuple[str | None, str | None]:
        warranty_text = "\n".join(
            self._strip_text(
                response.css("section.ui-pdp-warranty__section *::text, div.ui-pdp-warranty *::text").getall()
            )
        )
        returns_text = "\n".join(
            self._strip_text(
                response.css("section.ui-vpp-highlighted-specs__subtitles span::text, div.ui-pdp-return-policy *::text").getall()
            )
        )
        return warranty_text or None, returns_text or None

    def _get_attributes(self, response: Response) -> dict[str, str]:
        attributes: dict[str, str] = {}
        rows = response.xpath("//tr[contains(@class, 'andes-table__row') or ancestor::table[contains(@class,'spec')]]")
        if not rows:
            rows = response.xpath("//tr")
        for row in rows:
            label_parts = row.xpath(".//th//text() | .//span[contains(@class,'label')]//text()").getall()
            value_parts = row.xpath(".//td//text() | .//span[contains(@class,'value')]//text()").getall()
            label = " ".join(self._strip_text(label_parts)).lower()
            value = " ".join(self._strip_text(value_parts))
            if label:
                normalized = label.replace(":", "").strip()
                attributes.setdefault(normalized, value or None)

        if not attributes:
            items = response.css("li.ui-pdp-list__item")
            for item in items:
                label = " ".join(self._strip_text(item.css("strong::text").getall())).lower()
                value_text = " ".join(self._strip_text(item.css("span::text").getall()))
                if label:
                    attributes.setdefault(label.replace(":", "").strip(), value_text or None)

        return attributes

    # ------------------------------------------------------------------
    # Entrypoint
    # ------------------------------------------------------------------
    @staticmethod
    def run_spider(search_query: str | None = None, **spider_kwargs):
        os.makedirs(save_path, exist_ok=True)

        listing_path = os.path.join(save_path, "data.json")
        details_path = os.path.join(save_path, "product_details.json")

        process = CrawlerProcess(
            settings={
                **get_project_settings(),
                "FEEDS": {
                    listing_path: {
                        "format": "json",
                        "overwrite": True,
                        "item_classes": ["extraction.items.ListingItem"],
                    },
                    details_path: {
                        "format": "json",
                        "overwrite": True,
                        "item_classes": ["extraction.items.ProductDetailItem"],
                    },
                },
            }
        )
        spider_kwargs["max_pages"] = resolve_max_pages(cli_value=spider_kwargs.get("max_pages"))
        process.crawl(MercadoLivreSpider, search_query=search_query, **spider_kwargs)
        process.start()

    def closed(self, reason: str) -> None:  # type: ignore[override]
        self.logger.info(
            "Scraping finalizado (reason=%s): max_pages_requested=%s, pages_fetched=%s",
            reason,
            self.max_pages_requested,
            self.pages_fetched,
        )