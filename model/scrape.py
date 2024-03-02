import os
import re
import json
import requests
from urllib import request
from bs4 import BeautifulSoup as soup
from pydotmap import DotMap

furniture_styles = ["modern", "rustic", "retro", "vintage", "scandinavian", "colonial", "minimalist", "shabby chic", "traditional", "boho", "industrial", "art deco", "coastal", "midcentury modern"] 
furnitures = ["bedframe", "bookshelf", "chair", "nightstand", "ottoman", "shelves", "desks", "sofas", "dining tables", "coffee_tables", "dressers", "lamps", "plants"]

class PinterestImageScraper:

    def get_pinterest_links(body, max_images):
        searched_urls = []
        html = soup(body, 'html.parser')
        links = html.select('#main > div > div > div > a')
        for link in links:
            link = link.get('href')
            link = re.sub(r'/url\?q=', '', link)
            if link[0] != "/" and "pinterest" in link:
                if link not in searched_urls:
                    searched_urls.append(link)
                    if len(searched_urls) == max_images:
                        return searched_urls
                
        return searched_urls
    def save_image_url(self, max_images, pin_data):
        url_list = []
        for js in pin_data:
            data = DotMap(json.loads(js))
            urls = []
            for pin in data.props.initialReduxState.pins:
                if isinstance(data.props.initialReduxState.pins[pin].images.get("orig"), list):
                    for i in data.props.initialReduxState.pins[pin].images.get("orig"):
                        urls.append(i.get("url"))
                else:
                    urls.append(data.props.initialReduxState.pins[pin].images.get("orig").get("url"))
                for url in urls:
                    if url not in url_list:
                        if len(url_list) <= max_images:
                            url_list.append(url)
        return url_list

    def start_scraping(max_images, key):
        keyword = key + " pinterest"
        keyword = keyword.replace("+", "%20")
        url = f'http://www.google.com/search?hl=en&q={keyword}'
        res = requests.get(url)
        searched_urls = PinterestImageScraper.get_pinterest_links(res.content, max_images)

        return searched_urls


    def scrape(self, key, max_images):
        extracted_urls = PinterestImageScraper.start_scraping(max_images,key)
        data_list = []
        
        for url in extracted_urls:
            res = requests.get(url)
            html = soup(res.text, 'html.parser')
            json_data = html.find_all("script", attrs={"id": "__PWS_DATA__"})

            for a in json_data:
                data_list.append(a.string)

            url_list = self.save_image_url(max_images, data_list)
        return url_list

if __name__ == "__main__":
    scraper = PinterestImageScraper()
    for piece in furnitures:
        #for style in furniture_styles:
            fpath = rf'data\{piece}'
            exists = os.path.exists(fpath)
            if not exists:
                os.makedirs(fpath)
                urls = scraper.scrape(key=f'{piece}', max_images=250)
                for url in urls:
                    fname = os.path.join(fpath, url[40:])
                    request.urlretrieve(url, fname)