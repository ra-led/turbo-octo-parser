from pathlib import Path

from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
from docling.document_converter import DocumentConverter

IMAGE_RESOLUTION_SCALE = 2.0
ALLOWED_FORMATS = ['docx', 'pptx', 'xlsx']


def run_pipeline(data):
    input_doc_path = Path(data)

    doc_converter = DocumentConverter()
    conv_res = doc_converter.convert(input_doc_path)

    return conv_res.document.export_to_markdown()
