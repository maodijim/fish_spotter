"""
Script to download fish images from visfotos for image training. It scrapes the specified number of pages, finds image URLs, and saves them to a local folder.
"""

import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

def download_fish_images(base_url, target_folder, max_pages=5):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"Created folder: {target_folder}")

    current_url = base_url
    page_count = 0

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    while current_url and page_count < max_pages:
        print(f"\n--- Scraping page {page_count + 1}: {current_url} ---")
        try:
            response = requests.get(current_url, headers=headers, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching page {current_url}: {e}")
            break

        soup = BeautifulSoup(response.text, 'html.parser')

        img_tags = soup.find_all('img')

        download_count = 0
        skip_count = 0
        for img in img_tags:
            img_url = img.get('src')
            if not img_url:
                continue

            img_url = urljoin(current_url, img_url)

            if "/wp-content/uploads/" not in img_url or "favicon" in img_url:
                continue

            filename = os.path.basename(urlparse(img_url).path)
            if not filename or not any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                continue

            target_path = os.path.join(target_folder, filename)

            if os.path.exists(target_path):
                skip_count += 1
                continue

            try:
                print(f"Downloading: {filename}")
                img_response = requests.get(img_url, headers=headers, timeout=10)
                img_response.raise_for_status()
                with open(target_path, 'wb') as f:
                    f.write(img_response.content)
                download_count += 1
            except Exception as e:
                print(f"Failed to download {img_url}: {e}")

        print(f"Results: {download_count} downloaded, {skip_count} skipped.")

        # More robust next page finding
        next_page = None
        for a in soup.find_all('a'):
            if "Next page" in a.get_text():
                next_page = a
                break

        if next_page:
            current_url = urljoin(current_url, next_page.get('href'))
            page_count += 1
        else:
            print("No more 'Next page' link found.")
            break

if __name__ == "__main__":
    BASE_URL = "https://visdeurbel.nl/en/visfotos/"
    FOLDER = "fish_photos"
    download_fish_images(BASE_URL, FOLDER, max_pages=10)
