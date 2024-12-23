from pathlib import Path
# import pandas as pd
from docling_core.types.doc.document import (
    TextItem, PictureItem, SectionHeaderItem, TableItem, ListItem)
from PIL import Image, ImageDraw

import cv2
import numpy as np

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TableFormerMode
)
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    OcrMacOptions,
    # PdfPipelineOptions,
    RapidOcrOptions,
    TesseractCliOcrOptions,
    TesseractOcrOptions,
)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from collections import defaultdict
import json

# to explicitly prefetch:
artifacts_path = StandardPdfPipeline.download_models_hf()


class PipelineStep:
    def process(self, data):
        raise NotImplementedError("Subclasses should implement this!")


class LayoutAnalysisStep(PipelineStep):
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    FORMULA = "formula"
    LIST_ITEM = "list_item"
    PAGE_FOOTER = "page_footer"
    PAGE_HEADER = "page_header"
    PICTURE = "picture"
    SECTION_HEADER = "section_header"
    TABLE = "table"
    TEXT = "text"
    TITLE = "title"
    DOCUMENT_INDEX = "document_index"
    CODE = "code"
    CHECKBOX_SELECTED = "checkbox_selected"
    CHECKBOX_UNSELECTED = "checkbox_unselected"
    FORM = "form"
    KEY_VALUE_REGION = "key_value_region"

    def __init__(self):
        pipeline_options = PdfPipelineOptions(
                do_table_structure=True,
                artifacts_path=artifacts_path)
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE  # use more accurate TableFormer model
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.images_scale = 1.0
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True
        pipeline_options.ocr_options = EasyOcrOptions(lang=['ru', 'en'])

        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

    def process(self, data):
        conv_result = self.doc_converter.convert(data) # previously `convert_single`

        images_dir = Path("tmp/images")
        images_dir.mkdir(parents=True, exist_ok=True)

        pages_dir = Path("tmp/pages")
        pages_dir.mkdir(parents=True, exist_ok=True)

        # Save page images
        for page_no, page in conv_result.document.pages.items():
            page_no = page.page_no
            page_image_filename = pages_dir / f"{page_no}.png"
            with page_image_filename.open("wb") as fp:
                page.image.pil_image.save(fp, format="PNG")

        picture_counter = 0
        annotations = []
        idx = 1
        ## Iterate the elements in reading order, including hierachy level:
        for item, level in conv_result.document.iterate_items():
            if item.label in [self.PAGE_FOOTER, self.PAGE_HEADER]:
                continue
            prov_item = item.prov[0]
            # draw_bbox(pages_dir / f"{prov_item.page_no}.png", prov_item.bbox, item.label.lower())
            # print(level, type(item))
            image_pth = pages_dir / f"{prov_item.page_no}.png"
            bbox = prov_item.bbox
            image = Image.open(image_pth)
            x1, y1, x2, y2 = bbox.l, image.size[1] - bbox.t, bbox.r, image.size[1] - bbox.b
            item_image = image.crop((x1, y1, x2, y2))
            item_data = {
                'id': idx,
                'bbox': [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1],
                'page': prov_item.page_no
            }
            if not all(item_image.size):
                continue
            if isinstance(item, TextItem):
                print(item.text)
                fontsize, thickness = self.get_font_style(item_image, min_char_area=10)
                # print(f"Estimated font size: {fontsize}")
                # print(f"Estimated thickness: {thickness}")
                content = {
                    'text': item.text,
                    'type': 'h' if item.label.lower() in [self.SECTION_HEADER, self.TITLE] else 'p',
                    'fontsize': fontsize,
                    'thickness': thickness
                }
            elif isinstance(item, TableItem) and not isinstance(item, ListItem):
                # table_df: pd.DataFrame = item.export_to_dataframe()
                # print(table_df.to_markdown())
                # print(item.export_to_html())
                content = {
                    'html': item.export_to_html(),
                    'type': 'table'
                }
            elif isinstance(item, PictureItem):
                picture_counter += 1
                element_image_filename = images_dir / f"picture-{picture_counter}.png"
                with element_image_filename.open("wb") as fp:
                    item.get_image(conv_result.document).save(fp, "PNG")
                content = {
                    'src': str(element_image_filename),
                    'type': 'img'
                }
            elif isinstance(item, ListItem):
                # print(item.export_to_html())
                content = {
                    'text': item.text,
                    'type': 'li',
                    'fontsize': fontsize,
                    'thickness': thickness
                }
            item_data.update(content)
            annotations.append(item_data)
            idx += 1

        return annotations

    def get_thickness(self, binary):
        width = (binary > 128).sum(1)
        count = (np.diff(binary, axis=1) > 128).sum(1)
        thickness = width[count > 0] / count[count > 0]
        return np.median(thickness[~np.isnan(thickness)])

    def get_font_style(self, image, min_char_area=10):
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(gray.copy(), 128, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

        bounding_boxes = [bbox for bbox in bounding_boxes if bbox[2] * bbox[3] > min_char_area]

        heights = [bbox[3] for bbox in bounding_boxes]
        if len(heights) > 1:
            fontsize = np.quantile(heights, 0.5)
            thickness = self.get_thickness(binary)
        else:
            fontsize = None
            thickness = None
        return fontsize, thickness

    def draw_bbox(self, image_pth, bbox, title):
        color = 'blue'
        if title in [self.SECTION_HEADER, self.TITLE]:
            color = 'green'
        elif title in [self.TABLE, self.PICTURE]:
            color = 'red'
        elif title == self.CAPTION:
            color = 'orange'
        # draw boxes
        image = Image.open(image_pth)
        draw = ImageDraw.Draw(image)
        x1, y1  = bbox.l, image.size[1] - bbox.t
        x2, y2 = bbox.r, image.size[1] - bbox.b
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        draw.text((x1, y1), title, fill="red")
        image.save(image_pth)



class PDFToTextStep(PipelineStep):
    def process(self, data):
        # Dummy implementation
        return "Dummy text extracted from PDF"

class TextCleaningStep(PipelineStep):
    def process(self, data):
        # Dummy implementation
        return data.strip()

class TextToMarkdownStep(PipelineStep):
    def header_clusters(self, headers_info):
        # Extract features
        data = np.array([[item['fontsize'], item['thickness']] for item in headers_info])

        # Standardize the data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        # DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=2)
        labels = dbscan.fit_predict(data_scaled)

        # Assign cluster labels to headers
        for i, item in enumerate(headers_info):
            item['cluster'] = labels[i]

        # Assign cluster labels to headers
        for i, item in enumerate(headers_info):
            item['cluster'] = labels[i]

        # Group headers by clusters and compute mean values
        clusters = defaultdict(list)
        for item in headers_info:
            if item['cluster'] != -1:
                clusters[item['cluster']].append(item)

        cluster_stats = []
        for cluster_id, items in clusters.items():
            mean_fontsize = np.mean([item['fontsize'] for item in items])
            mean_thickness = np.mean([item['thickness'] for item in items])
            cluster_stats.append({
                'cluster_id': cluster_id,
                'mean_fontsize': mean_fontsize,
                'mean_thickness': mean_thickness,
                'items': items
            })

        # Sort clusters based on mean fontsize and thickness
        cluster_stats_sorted = sorted(
            cluster_stats,
            key=lambda x: (x['mean_fontsize'], x['mean_thickness']),
            reverse=True
        )
        return cluster_stats_sorted


    def convert_annotations_to_markdown(self, annotations):
        # Parse the JSON annotations
        items = annotations

        # Collect font sizes and thicknesses of headers
        headers_info = []
        for item in items:
            if item.get('type') == 'h':
                headers_info.append({
                    'id': item['id'],
                    'fontsize': item.get('fontsize', 0),
                    'thickness': item.get('thickness', 0),
                })
        # Headers clustering
        cluster_stats_sorted = self.header_clusters(headers_info)
        # Map clusters order to MD level
        header_level_map = {
            idx: i + 1 for i, stat in enumerate(cluster_stats_sorted)
            for idx in [item['id'] for item in stat['items']]
        }
        # Build the Markdown text
        markdown_lines = []
        for item in items:
            item_type = item['type']
            if item_type == 'h':
                # Get the header level
                header_level = header_level_map.get(item['id'], 1)
                print(header_level)
                prefix = '#' * header_level  # Markdown header prefix
                markdown_lines.append(f"{prefix} {item.get('text', '')}")
                print(f"{prefix} {item.get('text', '')}")
            elif item_type == 'p':
                markdown_lines.append(item.get('text', ''))
            elif item_type == 'li':
                markdown_lines.append(f"- {item.get('text', '')}")
            elif item_type == 'img':
                src = item.get('src', '')
                markdown_lines.append(f"![Image]({src})")
            elif item_type == 'table':
                html = item.get('html', '')
                # Include HTML directly
                markdown_lines.append(html)
            else:
                # If  type is unrecognized, skip it
                pass
            # Add empty line after each item for Markdown formatting
            markdown_lines.append('')

        # Join all lines into the final Markdown text
        markdown_text = '\n'.join(markdown_lines)
        return markdown_text


    def process(self, data):
        return self.convert_annotations_to_markdown(data)


def run_pipeline(pdf_path):
    data = pdf_path  # Starting data is the path to the PDF

    steps = [
        LayoutAnalysisStep(),
        #PDFToTextStep(),
        #TextCleaningStep(),
        TextToMarkdownStep(),
    ]

    for step in steps:
        data = step.process(data)

    return data  # Final Markdown content

