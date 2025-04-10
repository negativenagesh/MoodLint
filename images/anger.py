import requests
import urllib.parse
import os
import time

def scrape_unsplash_images(search_term, page=1, per_page=30, fetch_all=False, download_dir=None):
    """
    Fetch images from Unsplash API and optionally download them
    
    Args:
        search_term (str): The search query
        page (int): Page number (used only if fetch_all=False)
        per_page (int): Number of images per page (max 30)
        fetch_all (bool): If True, fetch all images across all pages
        download_dir (str): Directory to download images to
    """
    url = "https://unsplash.com/napi/search/photos"
    
    # Create download directory if it doesn't exist
    if download_dir and not os.path.exists(download_dir):
        os.makedirs(download_dir)
        print(f"Created directory: {download_dir}")
    
    if fetch_all:
        all_images = []
        current_page = 1
        while True:
            try:
                params = {
                    "page": current_page,
                    "per_page": min(per_page, 30),
                    "query": urllib.parse.quote(search_term.strip())
                }
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                results = data.get("results", [])
                
                if not results:
                    break
                
                for result in results:
                    image = {
                        "id": result.get("id", "unknown"),
                        "display_url": result["urls"].get("regular", result["urls"]["small"]),
                        "download_url": result["urls"].get("full", result["urls"]["regular"]),
                        "alt_text": result.get("alt_description", result.get("description", "Unsplash Image")),
                        "author_name": result["user"].get("name", "Unknown"),
                        "author_username": result["user"].get("username", "unknown"),
                        "height": result.get("height", 0),
                        "width": result.get("width", 0),
                        "created_at": result.get("created_at", "Unknown"),
                        "likes": result.get("likes", 0),
                        "color": result.get("color", "#000000")
                    }
                    all_images.append(image)
                    
                    # Download the image if directory is specified
                    if download_dir:
                        download_image(image, download_dir)
                
                current_page += 1
                time.sleep(0.5)  # Avoid rate limiting
            except Exception as e:
                print(f"Error fetching page {current_page}: {e}")
                break
        return all_images
    
    else:
        try:
            params = {
                "page": page,
                "per_page": min(per_page, 30),
                "query": urllib.parse.quote(search_term.strip())
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            images = []
            for result in data.get("results", []):
                image = {
                    "id": result.get("id", "unknown"),
                    "display_url": result["urls"].get("regular", result["urls"]["small"]),
                    "download_url": result["urls"].get("full", result["urls"]["regular"]),
                    "alt_text": result.get("alt_description", result.get("description", "Unsplash Image")),
                    "author_name": result["user"].get("name", "Unknown"),
                    "author_username": result["user"].get("username", "unknown"),
                    "height": result.get("height", 0),
                    "width": result.get("width", 0),
                    "created_at": result.get("created_at", "Unknown"),
                    "likes": result.get("likes", 0),
                    "color": result.get("color", "#000000")
                }
                images.append(image)
                
                # Download the image if directory is specified
                if download_dir:
                    download_image(image, download_dir)
                    
            total_pages = data.get("total_pages", 1)
            return images, total_pages
        except Exception as e:
            print(f"Error fetching images: {e}")
            return [], 1

def download_image(image, download_dir):
    """Download an image and save it to the specified directory"""
    try:
        image_url = image["download_url"]
        image_id = image["id"]
        filename = f"{image_id}.jpg"
        filepath = os.path.join(download_dir, filename)
        
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        
        with open(filepath, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        
        print(f"Downloaded: {filename}")
        
    except Exception as e:
        print(f"Error downloading image {image.get('id', 'unknown')}: {e}")

if __name__ == "__main__":
    # Create anger directory and download images
    anger_dir = "anger"
    scrape_unsplash_images("anger", per_page=30, download_dir=anger_dir)
    print(f"Finished downloading images to {anger_dir} directory")