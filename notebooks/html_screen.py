# Python version       : 3.9.14
# numpy   : 1.26.4
# PIL     : 10.4.0
# selenium: 4.4.3
# json    : 2.0.9
# goose3  : 3.1.19
import argparse
import json
import io
from goose3 import Goose
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException
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
PAGES_COUNT = 2
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
# service = Service(
#     '/Users/yuratomakov/.wdm/drivers/chromedriver/'
#     'mac64/131.0.6778.108/chromedriver-mac-x64/chromedriver'
# )
# driver = webdriver.Chrome(service=service, options=chrome_options)
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

# Get the size of the browser window
window_size = driver.get_window_size()
print(f"Window size: {window_size}")

# =============================================================================
# FIND MAIN CONTENT TAG
# =============================================================================
def get_content_element(driver):
    # try find <article>
    try:
        element = driver.find_element(By.TAG_NAME, "article")
        return element
    except NoSuchElementException:
        pass
    # try find <main>
    try:
        element = driver.find_element(By.TAG_NAME, "main")
        return element
    except NoSuchElementException:
        pass
        # auto detect content tag
    with Goose(goose_config) as g:
        article = g.extract(url=url)
        top_node = article.top_node
        top_node_tag = top_node.tag or ''
        top_node_class = top_node.get('class') or ''
        if top_node_class:
            top_node_class = '.' + '.'.join(top_node_class.split())
    element = driver.find_element(By.CSS_SELECTOR, f'{top_node_tag + top_node_class}')
    return element


element = get_content_element(driver)
element_location = element.location
element_size = element.size

# Calculate bounding box coordinates
x = element_location['x']
y = element_location['y']
width = element_size['width']
height = element_size['height']

print(f"Bounding box: ({x}, {y}, {x + width}, {y + height})")
# =============================================================================
# CROP CONTENT
# =============================================================================
PADDING = 35 # padding for text annotations
main_xl = x
main_yt = y
main_xr = min(W, x + width) + PADDING
main_yb = min(H, y + height) + PADDING

content_image = screenshot_image.crop((main_xl, main_yt, main_xr, main_yb))
content_image.save(prefix + "image.png")
content_draw = ImageDraw.Draw(content_image)

# # Create a drawing object
screenshot_draw = ImageDraw.Draw(screenshot_image)

# Draw the bounding box
screenshot_draw.rectangle([x, y, x + width, y + height], outline="red", width=2)

# Add annotation text
text = "Main content"
screenshot_draw.text((x, y - 10), text, fill="red")

# =============================================================================
# PDF SPANS LIKE TAGS
# =============================================================================
font = ImageFont.load_default(size=18)
span_tags = ['h1', 'h2', 'h3', 'table', 'li', 'img', 'p']
idx = 0
for tag in span_tags:
    elements = driver.find_elements(By.TAG_NAME, tag)
    for element in elements:
        idx += 1
        element_location = element.location
        element_size = element.size
    
        x = element_location['x']
        y = element_location['y']
        width = element_size['width']
        height = element_size['height']
        
        xl, yt, xr, yb = x, y, x + width, y + height
        if xl < main_xl or xr > main_xr or yt < main_yt or yb > main_yb:
            continue
    
        # Draw bounding box and annotation
        screenshot_draw.rectangle([x, y, x + width, y + height], outline="red", width=2)
        screenshot_draw.text((x, y - 10), tag.upper(), fill="red")
        screenshot_draw.text((x + width + 5, y + height // 2), f'#{idx}', fill="red", font=font)
        
        xl -= main_xl
        xr -= main_xl
        yt -= main_yt
        yb -= main_yt
        content_draw.rectangle([xl, yt, xr, yb], outline="red", width=2)
        content_draw.text((xl, yt - 10), tag.upper(), fill="red")
        content_draw.text((xr + 5, (yt + yb) // 2), f'#{idx}', fill="red", font=font)
    
        # Save annotation data
        bbox = (xl + xr) / 2, (yt + yb) / 2, xr - xl, yb - yt
        annotations.append({
            "bbox": bbox,
            "label": tag,
            "text": element.text,
            "id": idx
        })

# =============================================================================
# NMS
# =============================================================================
tag_priority = {tag: idx for idx, tag in enumerate(span_tags)}


def calculate_iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2
    union_area = bbox1_area + bbox2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def non_max_suppression(bboxes, iou_threshold=0.95):
    bboxes = sorted(bboxes, key=lambda x: (tag_priority[x['label']], -x['bbox'][2] * x['bbox'][3]))
    keep = []
    while bboxes:
        bbox = bboxes.pop(0)
        keep.append(bbox)
        bboxes = [b for b in bboxes if calculate_iou(bbox['bbox'], b['bbox']) < iou_threshold]
    return keep


print(f'Collected {len(annotations)}')
annotations = non_max_suppression(annotations)
print(f'Keeped {len(annotations)} after NMS')

# Save annotations to JSON
with open(prefix + "annotations.json", "w") as f:
    json.dump(annotations, f)

# =============================================================================
# GRAPH
# =============================================================================
INDENT_THRESHOLD = 8  # pixels

# Preprocess annotations: calculate left, top, right, bottom for each
for annotation in annotations:
    xc, yc, w, h = annotation['bbox']
    left = xc - w / 2
    top = yc - h / 2
    annotation['left'] = left
    annotation['top'] = top

# Sort annotations by 'top' coordinate (vertical position)
annotations.sort(key=lambda ann: ann['top'])

# Initialize data structures
heading_stack = []  # To keep track of heading hierarchy
li_indent_levels = []  # To map left indents to levels
last_p_or_h = None  # Last 'p' or heading encountered
processed_annotations = []  # To store annotations with their parents

# Build the directed graph
for annotation in annotations:
    tag = annotation['label']
    annotation['children'] = []  # Initialize children list

    if tag in ['h1', 'h2', 'h3']:
        # Determine heading level
        level = int(tag[1])

        # Adjust the heading stack
        while heading_stack and heading_stack[-1]['level'] >= level:
            heading_stack.pop()

        # Assign parent
        if heading_stack:
            parent_annotation = heading_stack[-1]['annotation']
            parent_annotation['children'].append(annotation)
            annotation['parent'] = parent_annotation
        else:
            annotation['parent'] = None  # Root level

        # Push current heading onto the stack
        heading_stack.append({'annotation': annotation, 'level': level})
        last_p_or_h = annotation  # Update last 'p' or heading

    elif tag == 'p':
        # Parent is the last heading in the stack if it exists
        if heading_stack:
            parent_annotation = heading_stack[-1]['annotation']
            parent_annotation['children'].append(annotation)
            annotation['parent'] = parent_annotation
        else:
            annotation['parent'] = None  # Orphan paragraph

        last_p_or_h = annotation  # Update last 'p' or heading

    elif tag in ['img', 'table']:
        # Parent is the last 'p' or heading
        if last_p_or_h:
            parent_annotation = last_p_or_h
            parent_annotation['children'].append(annotation)
            annotation['parent'] = parent_annotation
        else:
            annotation['parent'] = None  # Orphan image/table

    elif tag == 'li':

        current_left = annotation['left']

        # Determine the 'li' level based on indentation
        # Check if current left matches an existing indent level
        matched_level = None
        for idx, indent_level in enumerate(li_indent_levels):
            if abs(current_left - indent_level) <= INDENT_THRESHOLD:
                matched_level = idx
                break

        if matched_level is None:
            # New indent level
            li_indent_levels.append(current_left)
            li_indent_levels.sort()  # Keep indent levels sorted
            matched_level = li_indent_levels.index(current_left)

        level = matched_level + 1
        annotation['level'] = level

        # Find parent (last text element with higher level)
        for prev_annotation in reversed(processed_annotations):
            if prev_annotation['label'] in ['h1', 'h2', 'h3', 'p', 'li']:
                if prev_annotation['label'] == 'li':
                    if prev_annotation['level'] < level:
                        parent_annotation = prev_annotation
                        break
                else:
                    parent_annotation = prev_annotation
                    break
        else:
            parent_annotation = None  # No parent found

        if parent_annotation:
            parent_annotation['children'].append(annotation)
            annotation['parent'] = parent_annotation
        else:
            annotation['parent'] = None  # Orphan list item

    # Add the annotation to the processed list
    processed_annotations.append(annotation)


def print_hierarchy(annotation, indent=0):
    print('  ' * indent + f"{annotation['label'].upper()}: {annotation['text'].strip()[:35]}...")
    for child in annotation.get('children', []):
        with open(prefix + 'graph.txt', 'a') as f:
            f.write(f'#{annotation["id"]}->#{child["id"]}\n')
        parent_xy = (
            (annotation['bbox'][0] - annotation['bbox'][2] / 2),
            (annotation['bbox'][1] - annotation['bbox'][3] / 2)
        )
        child_xy = (child['bbox'][0], child['bbox'][1])
        content_draw.line([parent_xy, child_xy], fill='blue', width=3)
        content_draw.circle(child_xy, 3, fill='blue', width=3)
        print_hierarchy(child, indent + 1)

# Print the hierarchy starting from root nodes (annotations without parents)
# and store relations
with open(prefix + 'graph.txt', 'w') as f:
    f.write('')

for annotation in processed_annotations:
    if annotation['parent'] is None:
        print_hierarchy(annotation)

# =============================================================================
# STORE
# =============================================================================
# Save the annotated image
screenshot_image.save(prefix + "screen_image.png")
content_image.save(prefix + "annotated_image.png")
    
# Close the browser
driver.quit()

# import goose3
# import watermark.watermark as watermark
# print(watermark(iversions=True, globals_=globals()))
