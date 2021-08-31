from collections import defaultdict
from bs4 import BeautifulSoup
import requests
import string
import time
import json
import os
import re

punc = string.punctuation + "“”‘’"
punc = punc.replace(".", "")

def split_to_chars(s):
    return [e for e in s]

escape = split_to_chars(punc)

# Clean the path name
def clean_path(path:str):
    new_path = path.strip()
    new_path = ''.join([e if e not in escape else "_" for e in new_path])
    new_path = re.sub(r'\s', '_', new_path)
    return new_path



## Get all recipes available in this sites
def get_recipes():

    # Soupify the web page
    main_page = requests.get('https://veenaazmanov.com/recipes-in-text')
    html_main = main_page.content
    soup = BeautifulSoup(html_main, 'html.parser')

    # Find the recipe
    all_recipe = soup.find_all('a', target='_self')
    print(f'Tags with self targeted found: {len(all_recipe)}')

    all_recipe = all_recipe[49:]   # First 49 urls are not recipe

    # Grab url from href
    all_recipe_url = [a.get('href') for a in all_recipe]  

    # Grab the menu name
    menus = [a.string for a in all_recipe]   

    return all_recipe_url, menus


def get_menu_name(soup):
    try:
        menu = soup.find("h2", class_="wprm-recipe-name wprm-block-text-bold")
        if menu is None : raise AttributeError
    except AttributeError as e:
        non_menu_url.append(url)
        return soup, None
    return soup, menu.string


def get_recipe_instructions(soup):
    steps = [a for a in soup.find_all("div", class_="wprm-recipe-instruction-text")]
    return soup, steps


def append_images_url(soup, menu):
    key = clean_path(menu)
    content = soup.find_all('main', class_="content")
    
    try: 
        assert len(content) == 1
    except AssertionError as e:
        content_err[key] = url
    content = content[0]
    imgs = content.find_all('figure', class_="wp-block-image size-full")
    if len(imgs) == 0:
        imgs = content.find_all('figure', class_="wp-block-image")
        if len(imgs) == 0:
            img_not_detected[key] = url
            return

    for img in imgs:
        img_tag = img.img
        if (
            not img_tag['alt'].lower().startswith("ingredients") and 
            not img_tag['alt'].lower().startswith("progress")
        ):
            try:
                menu_imgs[key].append(img_tag['data-lazy-src'])
            except KeyError as e:
                menu_imgs[key].append(img_tag['src'])
            except KeyError as e:
                raise KeyError

## Make new directory if not exists
if not os.path.exists('images'): os.mkdir('images')


if __name__ == '__main__':
    # Get all menus name and their urls
    all_recipe_url, menus = get_recipes()

    unique_url = list(set(all_recipe_url))  # To make sure there is no duplicated url
    with open('unique_url.txt','w') as f:
        f.write('\n'.join(unique_url))

    data = {}          # Collect the output data
    non_menu_url = []  # Keep the failed url searching for recipe
    error_url = []     # Keep the failed request urls
    

    menu_imgs = defaultdict(lambda: []) # dict {menu: [images urls]}
    img_not_detected = {}               # urls failed to find image
    content_err = {}                    # error

    images = defaultdict(lambda: [])    # dicr {menu: [image.jpg, ...]}

    already_have_data = True if os.path.exists("data_strings_local.json") else False

    # Main loop for getting the data
    for url in unique_url:

        # Get each recipe page
        try:
            page = requests.get(url).content
            soup = BeautifulSoup(page, 'html.parser')
        except requests.exceptions.MissingSchema as e:
            error_url.append(url)
            continue

        # Find menu name
        soup, menu = get_menu_name(soup)
        if menu is None: continue
        menu = str(menu)
        
        # Clean the key for consistency
        key = clean_path(menu)

        # Find the recipe's instructions
        if not already_have_data:
            soup, steps = get_recipe_instructions(soup)
            data[key] = steps

        # Find images urls
        append_images_url(soup, menu)
        
        print(f"Data revieved: {len(data)} / Image url gathered {len(menu_imgs)}", end="\r")
        time.sleep(0.05)

    print(f"\n Data collected: {len(data)}")
    print(f"Non recipe url detected: {len(non_menu_url)}")
    print(f"Failed in get request: {len(error_url)}")
    print("="*50)


    n_imgs = 0
    n_menus = 0

    for menu, urls in menu_imgs.items():
        n_imgs+=len(urls)
        n_menus+=1

    print(f'Totally number of menus: {n_menus}')
    print(f'Totally collected images: {n_imgs}')

    ## This is final recipe data
    data_strings_new = {
        key: [str(step.text) for step in steps if step.text != None] for key, steps in data.items()
    }
    if not os.path.exists('data_strings_local.json'):
        json.dump(data_strings_new, open('data_strings_local.json', 'w'))

    ## This is resulting dict{menu:[image_urls]}
    json.dump(menu_imgs, open("menu_imgs_url.json","w"))


    ## Getting the images by iterating over their urls.
    for key, urls in list(menu_imgs.items()):
        
        for i, url in enumerate(urls):
            image = requests.get(url)   # Get the image
            
            path = f"images/{key}_{i}.jpg"
            images[key].append(path)    # Keep menu image's path
            with open(path, "wb") as f:
                f.write(image.content)
        
        print(f'Progress: {(len(images)/len(menu_imgs)):.2f}', end='\r')
        time.sleep(0.1)

    json.dump(dict(images), open("image_path.json","w"))
