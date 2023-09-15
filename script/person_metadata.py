import pandas as pd
import logging
import traceback
from bs4 import BeautifulSoup as BS
import requests
import sys
sys.path.insert(0, '..')
from DataCollectionHelper import MyLogger, SaveData
import re
import time

'''
Gets information for the LARGE!!! dataset of people (biography, quotes from interviews, pictures) 
to see if this information is useful for predicting race.

from /bio -> Overview (born, birth name, nicknames, height)
            Mini Bio
            Family
            Trade Mark
            Trivia
            Personal Quotes

'''


log_file = "person_metadata.log"
# HEADERS = {'User-Agent': 'Mozilla/5.0'}
HEADERS = {'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'}

def get_bio_metadata(person_href, person_name, save_to_file):
    '''
    Parses biography page from imdb /bio
    '''
    try:
        metadata = {"overview": None, "mini_bio": None, "family": None, "trademark": None, "trivia": None, "quotes": None}
        person_id = SaveData.extract_id_from_href(person_href, include_text=False)
        page_source = get_biography_page(person_id, person_name, save_to_file=save_to_file)
        bio_content = page_source.find(id="bio_content")
        headers = [header.get_text() for header in bio_content.find_all(class_="li_group")]

        for header in headers:
            header_item_count = re.search(r"\((.*)\)", header).group(1)
            if "Overview" in header:
                overview = get_overview(bio_content)
                metadata["overview"] = overview
            elif "Mini Bio" in header:
                mini_bio = get_mini_bio(bio_content)
                metadata["mini_bio"] = mini_bio
            elif "Family" in header:
                family = get_family(bio_content)
                metadata["family"] = family
            elif "Trade Mark" in header:
                trademark = get_trademark(bio_content, int(header_item_count))
                metadata["trademark"] = trademark
            elif "Trivia" in header:
                trivia = get_trivia(bio_content, int(header_item_count))
                metadata["trivia"] = trivia
            elif "Quotes" in header:
                quotes = get_quotes(bio_content, int(header_item_count))
                metadata["quotes"] = quotes
            # else:
            #     print(header,"\tMetadata not included")
        return metadata["overview"], metadata["mini_bio"], metadata["family"], metadata["trademark"], metadata["trivia"], metadata["quotes"]
    except Exception as e:
        print(f"Failed in get_bio_metadata() {person_name} \t {person_href} \t{e}")

def get_biography_page(person_id, person_name, save_to_file):
    try:
        person_url = f'https://www.imdb.com/name/nm{int(person_id):07}'
        bio_url = f'{person_url}/bio'
        response = requests.get(bio_url, headers=HEADERS)
        time.sleep(0.5)
        MyLogger.insert(log_file, f"Response {response.status_code} {person_name} \t {person_id}", logging.INFO)
        page_source = BS(response.text, 'html.parser')

        if save_to_file:
            save_biography_page(person_name, person_id, page_source)

        return page_source
    except Exception as e:
        print(f"Failed in get_biography_page() for {person_name}\t{person_id}\t{traceback.format_exc()}")
        MyLogger.insert(log_file, f"Failed in get_biography_page() {person_name} \t {person_id} \t{e}", logging.ERROR)
        return None

def save_biography_page(person_name, person_id, html):
    '''
    Output looks like: Ang Lee_487_bio.html
    '''
    with open(f"bios/{person_name}_{person_id}_bio.html", "w", encoding='utf-8') as file:
        file.write(str(html))

def get_overview(bio_content):
    try:
        overview = {}
        table = bio_content.find("table", id="overviewTable")
        rows = table.find_all("tr")
        for row in rows:
            key_value_pair = row.find_all("td")
            if len(key_value_pair) == 2:
                key, value = key_value_pair     
                key_text = " ".join(key.get_text().split()).replace("\xa0", " ")
                value_text = " ".join(value.get_text().split()).replace("\xa0", " ")
                overview[key_text] = value_text
            else:
                MyLogger.insert(log_file, f"Not a key,value pair in get_overview()", logging.WARNING)
        return overview
    except:
        MyLogger.insert(log_file, f"Did not properly obtain overview in get_overview()", logging.WARNING)
  
def get_mini_bio(bio_content):
    try:
        bio = bio_content.find("a", {"name":"mini_bio"})
        bio_text = bio.findNext("div").get_text()
        return bio_text
    except:
        MyLogger.insert(log_file, f"Did not properly obtain mini bio in get_mini_bio()", logging.WARNING)
  
def get_family(bio_content):
    try:
        overview = {}
        table = bio_content.find("table", id="tableFamily")
        rows = table.find_all("tr")
        for row in rows:
            info = row.find_all("td")
            if len(info) == 2:
                key, value = info
                key_text = " ".join(key.get_text().split()).replace("\xa0", " ")
                value_text = " ".join(value.get_text().split()).replace("\xa0", " ")
                overview[key_text] = value_text
            else:
                MyLogger.insert(log_file, f"Not a key,value pair in get_family()", logging.WARNING)
        return overview
    except:
        MyLogger.insert(log_file, f"Did not properly obtain family in get_family()", logging.WARNING)
       
def get_trademark(bio_content, count, max_count=10):
    try:
        trademark = bio_content.find("a", {"name":"trademark"})
        count = min(count, max_count)
        trademark_items = []

        i = 0
        while i < count:
            trademark = trademark.find_next_sibling("div")
            trademark_text = trademark.get_text().strip()
            trademark_items.append(trademark_text)
            i += 1

        return trademark_items
    except:
        MyLogger.insert(log_file, f"Did not properly obtain trademark in get_trademark()", logging.WARNING)
       
def get_trivia(bio_content, count, max_count=10):
    try:
        trivia = bio_content.find("a", {"name":"trivia"})
        count = min(count, max_count)
        trivia_items = []

        i = 0
        while i < count:
            trivia = trivia.find_next_sibling("div")
            trivia_text = trivia.get_text().strip()
            trivia_items.append(trivia_text)
            i += 1

        return trivia_items
    except:
        MyLogger.insert(log_file, f"Did not properly obtain trivia in get_trivia()", logging.WARNING)

def get_quotes(bio_content, count, max_count=10):
    try:
        quote = bio_content.find("a", {"name":"quotes"})
        count = min(count, max_count)
        quote_items = []

        i = 0
        while i < count:
            quote = quote.find_next_sibling("div")
            quote_text = quote.get_text().strip()
            quote_items.append(quote_text)
            i += 1

        return quote_items
    except:
        MyLogger.insert(log_file, f"Did not properly obtain quotes in get_quotes()", logging.WARNING)

def get_media_page(person_id, person_name):
    '''
    TODO: not sure whether this is necessary to use yet
    '''
    try:
        person_url = f'https://www.imdb.com/name/nm{int(person_id):07}'
        media_url = f'{person_url}/mediaindex'
        response = requests.get(media_url, headers=HEADERS)
        print(response)
        page_source = BS(response.text, 'html.parser')
        
        # get profile image
        profile = page_source.find("img", class_="poster", src=True)
        save_image(person_name, person_id, "IMDB Profile", profile["src"])
        
        # get other photos
        photo_grid = page_source.find(id="media_index_thumbnail_grid")
        photo_urls = photo_grid.find_all("img", src=True)
        print(len(photo_urls))
        for tag in photo_urls:
            title = tag["alt"]
            url = tag["src"]
            save_image(person_name, person_id, title, url)

    except Exception as e:
        print(f"Failed in get_media_page() for {person_name}\t{person_id}\t{traceback.format_exc()}")
        MyLogger.insert(log_file, f"Failed in get_media_page() {person_name} \t {person_id} \t{e}", logging.ERROR)

def save_image(person_name, person_id, title, image_url):
    '''
    Image file naming convention:
    e.g. Tom Cruise_129_{"Title of the Image"}.jpg
    
    '''
    try:
        response = requests.get(image_url)
        if response.status_code:
            file = open(f'images/{person_name}_{person_id}_"{title}".png', 'wb')
            file.write(response.content)
            file.close()
        else:
            MyLogger.insert(log_file, f"save_image()\tResponse code {response.status_code} for {person_name} \t {person_id} \t {image_url}", logging.WARNING)

    except Exception as e:
        print(f"Failed to save image in save_image() for {person_name}\t{person_id}\t{traceback.format_exc()}")
        MyLogger.insert(log_file, f"Failed to save image in save_image()  {person_name} \t {person_id} \t{e}", logging.ERROR)

############################# TESTS #################################
# cinque = "497046"
# tomcruise = "129"
# brian = "2265122"
# willsmith = "226"
# anglee = "487"

# get_biography_page(tomcruise, "Tom Cruise")
# get_media_page(tomcruise, "Tom Cruise")
# get_bio_metadata(tomcruise, "Tom Cruise")
# get_bio_metadata(brian, "Brian B")
# get_bio_metadata(cinque, "Cinque")
############################# MAIN #################################

def main():
    df = pd.read_csv("new_names_to_final_sample.csv")
    df["overview"],df["mini_bio"],df["family"],df["trademark"],df["trivia"],df["quotes"] = zip(*df.apply(lambda x: get_bio_metadata(x["href"], x["name"], save_to_file=False), axis=1))
    df.to_csv("new_names_to_final_sample_metadata.csv", mode="a")
    
# main()
