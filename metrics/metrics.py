"""
Подсчет метрик парсинга pdf-документов:

1. cd metrics
2. pip install -r requirements.txt
3. закинуть в папку test_data все тестовые документы вместе с разметкой
   (по аналогии с примером example.pdf и example.json)
4. python3 metrics.py --api-url <адрес АПИ конвертера>
"""

import argparse
import json
import os
import time
from typing import NamedTuple, List, Optional
from zipfile import ZipFile

import Levenshtein
import requests
from tqdm import tqdm


current_dir = os.path.dirname(__file__)
test_data_dir = os.path.join(current_dir, "test_data")


parser = argparse.ArgumentParser(description='Process an API URL for converting pdf to md')
parser.add_argument('--api-url', type=str, help='API URL of converter', required=True)
args = parser.parse_args()

API_URL = args.api_url
CONVERT_HANDLER = "/convert"
STATUS_HANDLER = "/status"
RESULT_HANDLER = "/result"


class Header(NamedTuple):
    """Раздел документа."""
    level: int
    text: str


class ParseResult(NamedTuple):
    """Результат парсинга pdf-документа."""
    tables: int
    images: int
    headers: List[Header]


class Metrics(NamedTuple):
    """Метрики парсинга pdf-документа."""
    found_headers_percentage: float
    right_hierarchy_headers_percentage: float
    found_tables_percentage: float
    found_images_percentage: float


def is_converting_completed(task_id: str) -> bool:
    """Проверка завершения процесса конвертации pdf в md."""
    r = requests.get(API_URL + STATUS_HANDLER, params={"task_id": task_id})
    try:
        status = r.json()["status"]
    except:
        return True
    else:
        if status == "completed":
            return True
        else:
            return False


def get_convert_result(task_id: str) -> bytes:
    """Получение результата конвертации (zip-архив)."""
    status_code = None
    while status_code != 200:
        print("Waiting for result is available")
        time.sleep(10)
        r = requests.get(API_URL + RESULT_HANDLER + "/" + task_id)
        status_code = r.status_code
    content = r.content
    return content


def get_md_content_from_zip_archive(zip_: bytes) -> str:
    """Вытаскивание текста md-файла из zip-архива."""
    zip_file = "result.zip"
    md_file = "text.md"
    with open(zip_file, 'wb') as fwb:
        fwb.write(zip_)
    with ZipFile(zip_file) as zip_f:
        with zip_f.open(md_file) as md_f:
            md_content = md_f.read().decode()
    os.remove(zip_file)
    return md_content


def convert_pdf(pdf_file: str) -> str:
    """На входе путь до файла на выходе содержимое полученного markdown-файла."""
    with open(pdf_file, "rb") as frb:
        multipart_form_data = {"file": frb}
        r = requests.post(API_URL + CONVERT_HANDLER, files=multipart_form_data)
    if r.status_code == 200:
        r_result = r.json()
        task_id = r_result["task_id"]
        print(f"Task id:\n{task_id}")
        is_converting_completed_ = False
        while not is_converting_completed_:
            print("Waiting for converting is finished")
            time.sleep(10)
            is_converting_completed_ = is_converting_completed(task_id)
        zip_bytes = get_convert_result(task_id)
        md_content = get_md_content_from_zip_archive(zip_bytes)
        return md_content
    else:
        return ""


def str_to_header(markdown_row: str) -> Header:
    assert markdown_row.startswith("#")
    text = markdown_row.lstrip("#")
    level = len(markdown_row) - len(text) - 1
    header = Header(level, text.strip())
    return header


def get_actual_result(md_content: str) -> ParseResult:
    """Получение фактического результата из полученного markdown-файла."""
    elements = [elem.strip() for elem in md_content.split("\n") if elem]
    tables = len([elem for elem in elements if elem.startswith("<table")])
    images = len([elem for elem in elements if elem.startswith("![")])
    headers = [str_to_header(elem) for elem in elements if elem.startswith("#")]
    parse_result = ParseResult(tables, images, headers)
    return parse_result


def get_expected_result(pdf_file: str) -> Optional[ParseResult]:
    """Получение ожидаемого результата для указанного pdf-документа."""
    json_file = pdf_file.rstrip(".pdf") + ".json"
    try:
        with open(json_file, "r", encoding="utf-8") as fr:
            try:
                expected_result_dict = json.load(fr)
            except Exception as err:
                print(json_file)
                raise(err)
    except FileNotFoundError:
        return None
    headers = [
        Header(level=elem["level"], text=elem["text"])
        for elem in expected_result_dict["headers"]
    ]
    expected_result = ParseResult(
        tables=expected_result_dict["tables"],
        images=expected_result_dict["images"],
        headers=headers,
    )
    return expected_result


def is_texts_similar(text1: str, text2: str) -> bool:
    """Являются ли тексты одинаковыми с учетом небольших опечаток."""
    SIMILARITY_THRESHOLD = 0.8
    return Levenshtein.ratio(text1, text2, processor=lambda x: x.strip().lower()) >= SIMILARITY_THRESHOLD


def count_quantity_metrics(expected_quantity: int, actual_quantity: int) -> float:
    """Метрика количества найденных таблиц или изображений."""
    if actual_quantity <= expected_quantity:
        result = actual_quantity/expected_quantity
    else:
        result = expected_quantity/actual_quantity
    return round(result, 2)


def find_expected_header_in_actual_headers(
    expected_header: Header, actual_headers: List[Header]
) -> Optional[Header]:
    """Поиск раздела из эталонной разметки среди разделов результата парсинга pdf-документа."""
    result = None
    for actual_header in actual_headers:
        if is_texts_similar(expected_header.text, actual_header.text):
            result = actual_header
            break
    return result


def count_found_headers_quantity(
    expected_headers: List[Header], actual_headers: List[Header]
) -> int:
    """Кол-во найденных разделов документа."""
    found_headers = 0
    for expected_header in expected_headers:
        if find_expected_header_in_actual_headers(expected_header, actual_headers) is not None:
            found_headers +=1
    return found_headers


def count_found_headers_percentage(
    expected_headers: List[Header], actual_headers: List[Header]
) -> float:
    """Доля найденных разделов документа."""
    total_headers = len(expected_headers)
    found_headers = count_found_headers_quantity(expected_headers, actual_headers)
    if total_headers != 0:
        result = found_headers / total_headers
    else:
        result = 1
    return round(result, 2)


def get_parent(header: Header, previous_headers: List[Header]) -> Optional[Header]:
    """Поиск родительского раздела."""
    header_level = header.level
    parent_header = None
    for prev_header in previous_headers[::-1]:
        if prev_header.level < header_level:
            parent_header = prev_header
            break
    return parent_header


def count_total_hierarchy_quantity(headers: List[Header]) -> int:
    """Подсчет кол-во отношений 'родитель -> дочерний элемент'."""
    counter = 0
    for idx in range(len(headers)):
        parent = get_parent(headers[idx], headers[:idx])
        if parent is not None:
            counter += 1
    return counter


def count_right_hierarchy_headers_percentage(
    expected_headers: List[Header], actual_headers: List[Header]
) -> float:
    """Доля верно определенных иерархий разделов документа."""
    total_hierarchy_quantity = 0
    right_hierarchy_headers_counter = 0
    for idx in range(len(expected_headers)):
        expected_header = expected_headers[idx]
        prev_expected_headers = expected_headers[:idx]
        expected_header_parent = get_parent(expected_header, prev_expected_headers)
        actual_header = find_expected_header_in_actual_headers(expected_header, actual_headers)
        if actual_header is not None:
            total_hierarchy_quantity += 1
            actual_header_idx = actual_headers.index(actual_header)
            prev_actual_headers = actual_headers[:actual_header_idx]
            actual_header_parent = get_parent(actual_header, prev_actual_headers)
            if actual_header_parent is None:
                if expected_header_parent is None:
                    right_hierarchy_headers_counter += 1
            else:
                if (
                    expected_header_parent is not None
                    and is_texts_similar(expected_header_parent.text, actual_header_parent.text)
                ):
                    right_hierarchy_headers_counter += 1
    if total_hierarchy_quantity != 0:
        result = right_hierarchy_headers_counter/total_hierarchy_quantity
    else:
        result = 0
    return round(result, 2)


def process_pdf_file(pdf_file: str) -> Optional[Metrics]:
    """Парсинг и подсчет метрик для pdf-файла."""
    print(f"File:\n{pdf_file}")
    expected_result = get_expected_result(pdf_file)
    if expected_result is None:
        return None
    md_content = convert_pdf(pdf_file)
    actual_result = get_actual_result(md_content)
    metrics = Metrics(
        found_headers_percentage=count_found_headers_percentage(
            expected_headers=expected_result.headers,
            actual_headers=actual_result.headers,
        ),
        right_hierarchy_headers_percentage=count_right_hierarchy_headers_percentage(
            expected_headers=expected_result.headers,
            actual_headers=actual_result.headers,
        ),
        found_tables_percentage=count_quantity_metrics(
            expected_quantity=expected_result.tables,
            actual_quantity=actual_result.tables,
        ),
        found_images_percentage=count_quantity_metrics(
            expected_quantity=expected_result.images,
            actual_quantity=actual_result.images,
        ),
    )
    return metrics


def pretty_print(metrics: dict) -> None:
    """Красивый вывод в консоль словаря с метриками."""
    print(json.dumps(metrics, ensure_ascii=False, indent=4))


def main() -> None:
    """Точка входа."""
    metrics_dict = dict()
    metrics_dict["total"] = dict()
    metrics_dict["total"]["Доля найденных разделов документа"] = 0
    metrics_dict["total"]["Доля верно определенных иерархий разделов документа"] = 0
    metrics_dict["total"]["Метрика найденных таблиц документа"] = 0
    metrics_dict["total"]["Метрика найденных изображений документа"] = 0

    pdf_files_in_test_data_dir = [file for file in os.listdir(test_data_dir) if file.endswith(".pdf")]
    for file in tqdm(pdf_files_in_test_data_dir):
        if file.endswith(".pdf"):
            pdf_file = os.path.join(test_data_dir, file)
            metrics_dict[pdf_file] = dict()
            pdf_metrics = process_pdf_file(pdf_file)
            if pdf_metrics is not None:
                metrics_dict[pdf_file]["Доля найденных разделов документа"] = pdf_metrics.found_headers_percentage
                metrics_dict[pdf_file]["Доля верно определенных иерархий разделов документа"] = pdf_metrics.right_hierarchy_headers_percentage
                metrics_dict[pdf_file]["Метрика найденных таблиц документа"] = pdf_metrics.found_tables_percentage
                metrics_dict[pdf_file]["Метрика найденных изображений документа"] = pdf_metrics.found_images_percentage

                metrics_dict["total"]["Доля найденных разделов документа"] = \
                    metrics_dict["total"]["Доля найденных разделов документа"] + pdf_metrics.found_headers_percentage
                metrics_dict["total"]["Доля верно определенных иерархий разделов документа"] = \
                    metrics_dict["total"]["Доля верно определенных иерархий разделов документа"] + pdf_metrics.right_hierarchy_headers_percentage
                metrics_dict["total"]["Метрика найденных таблиц документа"] = \
                    metrics_dict["total"]["Метрика найденных таблиц документа"] + pdf_metrics.found_tables_percentage
                metrics_dict["total"]["Метрика найденных изображений документа"] = \
                    metrics_dict["total"]["Метрика найденных изображений документа"] + pdf_metrics.found_images_percentage

    pdfs_quantity = len([item for item in metrics_dict.values() if item != {}]) - 1
    if pdfs_quantity != 0:
        metrics_dict["total"]["Доля найденных разделов документа"] = \
            metrics_dict["total"]["Доля найденных разделов документа"] / pdfs_quantity
        metrics_dict["total"]["Доля верно определенных иерархий разделов документа"] = \
            metrics_dict["total"]["Доля верно определенных иерархий разделов документа"] / pdfs_quantity
        metrics_dict["total"]["Метрика найденных таблиц документа"] = \
            metrics_dict["total"]["Метрика найденных таблиц документа"] / pdfs_quantity
        metrics_dict["total"]["Метрика найденных изображений документа"] = \
            metrics_dict["total"]["Метрика найденных изображений документа"] / pdfs_quantity
    else:
        metrics_dict["total"] = dict()
    pretty_print(metrics_dict)


if __name__ == "__main__":
    main()