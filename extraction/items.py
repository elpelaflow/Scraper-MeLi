+37
-4

# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class ListingItem(scrapy.Item):
    name = scrapy.Field()
    seller = scrapy.Field()
    price = scrapy.Field()
    reviews_rating_number = scrapy.Field()
    reviews_amount = scrapy.Field()
    product_url = scrapy.Field()
    item_id = scrapy.Field()


class ProductDetailItem(scrapy.Item):
    item_id = scrapy.Field()
    product_url = scrapy.Field()
    title_detail = scrapy.Field()
    price_detail = scrapy.Field()
    currency = scrapy.Field()
    condition = scrapy.Field()
    sold_quantity = scrapy.Field()
    brand = scrapy.Field()
    model = scrapy.Field()
    color = scrapy.Field()
    material = scrapy.Field()
    capacity = scrapy.Field()
    voltage = scrapy.Field()
    seller_name = scrapy.Field()
    seller_location = scrapy.Field()
    official_store_flag = scrapy.Field()
    shipping_full_flag = scrapy.Field()
    shipping_free_flag = scrapy.Field()
    rating_value = scrapy.Field()
    rating_count = scrapy.Field()
    description_plain = scrapy.Field()
    images_json = scrapy.Field()
    breadcrumbs_json = scrapy.Field()
    warranty_text = scrapy.Field()
    returns_text = scrapy.Field()
    scrap_date_detail = scrapy.Field()