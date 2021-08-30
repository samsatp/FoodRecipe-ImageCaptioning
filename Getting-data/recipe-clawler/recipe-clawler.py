from bs4 import BeautifulSoup
import requests
import time
import json
import sys
import os

## Get all recipes available in this sites
def get_recipes() -> (all_recipe_url, menus_name):

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


all_recipe_url, menus = get_recipes()

data = {}          # Collect the output data
non_menu_url = []  # Keep the failed url search
error_url = []     # Keep the failed url search
unique_url = list(set(all_recipe_url))  # To make sure there is no duplicated url

with open('unique_url.txt','w') as f:
    f.write('\n'.join(unique_url))


for url in unique_url:

    try:
        page = requests.get(url).content
    except requests.exceptions.MissingSchema as e:
        error_url.append(url)
        continue

    soup = BeautifulSoup(page, 'html.parser')
    
    try:
        menu = soup.find("h2", class_="wprm-recipe-name wprm-block-text-bold")
        if menu is None : raise AttributeError
    except AttributeError as e:
        non_menu_url.append(url)
        continue

    steps = [a for a in soup.find_all("div", class_="wprm-recipe-instruction-text")]
    data[str(menu.string)] = steps

    sys.stdout.write("\rData revieved: %i" % len(data))
    sys.stdout.flush()
    time.sleep(0.05)

print(f"Data collected: {len(data)}")
print(f"Non recipe url detected: {len(non_menu_url)}")
print(f"Failed in get request: {len(error_url)}")

data_strings_new = {
    key: [str(step.text) for step in steps if step.text != None] for key, steps in data.items()
}



if not os.path.exists('data_strings_local.json'):
    json.dump(data_strings_new, open('data_strings_local.json', 'w'))