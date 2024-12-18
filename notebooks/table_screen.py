import argparse
import json
import io
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from PIL import Image, ImageDraw, ImageFont

# =============================================================================
# COMMAND LINE ARGUMENTS
# =============================================================================
parser = argparse.ArgumentParser(description='Process a URL to extract content.')
parser.add_argument('--url', type=str, help='The URL of the web page to process')
parser.add_argument('--prefix', type=str, help='The results prefix', default='')
args = parser.parse_args()

url = args.url
prefix = args.prefix + '_' if args.prefix else ''
# =============================================================================
# CONFIG
# =============================================================================
# Screen size
A4_ASPECT_RATIO = 200 / 300 # W / H
PAGES_COUNT = 5
W = 1000
H = int(W / A4_ASPECT_RATIO * PAGES_COUNT)

# Goose config to extract readable content
goose_config = {}
goose_config['strict'] = False  # turn of strict exception handling
goose_config['browser_user_agent'] = 'Mozilla 5.0'  # set the browser agent string
goose_config['http_timeout'] = 5.05  # set http timeout in seconds

# Set up Selenium WebDriver
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument(f"--window-size={W},{H}")
driver = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)

# =============================================================================
# GET PAGE
# =============================================================================
driver.get(url)
annotations = []

# Capture the full page screenshot
screenshot = driver.get_screenshot_as_png()
screenshot_image = Image.open(io.BytesIO(screenshot))

screenshot_image.thumbnail((W, H), Image.Resampling.LANCZOS)

# =============================================================================
# PDF SPANS LIKE TAGS
# =============================================================================
font = ImageFont.load_default(size=18)
idx = 0

elements = driver.find_elements(By.TAG_NAME, 'table')
for element in elements:
    idx += 1
    element_location = element.location
    element_size = element.size

    x = element_location['x']
    y = element_location['y']
    width = element_size['width']
    height = element_size['height']

    table_image = screenshot_image.crop((x, y, x + width, y + height))
    table_image.save(prefix + f"table_{idx}.png")
    
    table_draw = ImageDraw.Draw(table_image)
    td_elements = element.find_elements(By.TAG_NAME, 'td')
    td_idx = 0
    for td_element in td_elements:
        td_idx += 1
        td_location = td_element.location
        td_size = td_element.size
        
        td_x = td_location['x'] - x
        td_y = td_location['y'] - y
        td_width = td_size['width']
        td_height = td_size['height']
        
        td_xl, td_yt, td_xr, td_yb = td_x, td_y, td_x + td_width, td_y + td_height

        # Draw bounding box and annotation for td
        table_draw.rectangle([td_x, td_y, td_x + td_width, td_y + td_height], outline="blue", width=2)
        table_draw.text((td_x, td_y - 10), 'TD', fill="blue")
        table_draw.text((td_x + td_width + 5, td_y + td_height // 2), f'#{td_idx}', fill="blue", font=font)

        # Save annotation data for td
        td_bbox = (td_xl + td_xr) / 2, (td_yt + td_yb) / 2, td_xr - td_xl, td_yb - td_yt
        annotations.append({
            "bbox": td_bbox,
            "label": 'td',
            "text": td_element.text,
            "id": td_idx,
            "table_id": idx
        })

    # Save the cropped table image with td annotations
    table_image.save(prefix + f"annotated_table_{idx}.png")

# =============================================================================
# STORE
# =============================================================================
with open(prefix + "annotations.json", "w") as f:
    json.dump(annotations, f)

# Close the browser
driver.quit()
