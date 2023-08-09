import os
import pytest
import httpx
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "request" in response.context

def test_convert_with_two_images():
    test_file_path1 = os.path.join(os.path.dirname(__file__), 'data', 'test_image00-00.jpeg')
    test_file_path2 = os.path.join(os.path.dirname(__file__), 'data', 'test_image01-00.jpeg')

    with open(test_file_path1, 'rb') as test_file1:
        files1 = {"images": ("test_image00-00.jpg", test_file1, 'image/jpeg')}
        response1 = client.post("/convert", files=files1)
    with open(test_file_path2, 'rb') as test_file2:
        files2 = {"images": ("test_image01-00.jpg", test_file2, 'image/jpeg')}
        response2 = client.post("/convert", files=files2)

    assert response1.status_code == 200
    assert response2.status_code == 200
def test_convert_with_no_images():
    files = {"images": []}
    response = client.post("/convert", files=files)

    assert response.status_code == 400

def test_evaluation_with_one_image_one_text_file():
    test_image_path = os.path.join(os.path.dirname(__file__), 'data', 'test_image00-00.jpeg')
    test_text_path = os.path.join(os.path.dirname(__file__), 'data', 'test_scan00-00.txt')

    with open(test_image_path, 'rb') as test_image, open(test_text_path, 'rb') as test_text:
        files = {
            "ocr_files": ("test_image00-00.jpg", test_image, 'image/jpeg'),
            "gt_files": ("test_text00-00.txt", test_text, 'text/plain')
        }
        response = client.post("/evaluation", files=files)

    assert response.status_code == 200


def test_evaluation_with_no_files():
    files = {"ocr_files": [], "gt_files": []}
    response = client.post("/evaluation", files=files)

    assert response.status_code == 400


@pytest.mark.skip(reason="no point in testing, takes too long")
def test_experiment_with_one_image_one_text_file():
    test_image_path = os.path.join(os.path.dirname(__file__), 'data', 'test_image00-00.jpeg')
    test_text_path = os.path.join(os.path.dirname(__file__), 'data', 'test_scan00-00.txt')

    with open(test_image_path, 'rb') as test_image, open(test_text_path, 'rb') as test_text:
        files = {
            "ocr_files": [("test_image00-00.jpg", test_image, 'image/jpeg')],
            "gt_files": [("test_scan00-00.txt", test_text, 'text/plain')]
        }
        with httpx.Client(timeout=500.0) as client:
            response = client.post("http://localhost:8000/experiment", files=files)
        print(response.content)
    assert response.status_code == 200

def test_experiment_with_no_files():
    files = {"ocr_files": [], "gt_files": []}
    response = client.post("/experiment", files=files)

    assert response.status_code == 400

def test_process_csv_wer():
    test_csv_path = os.path.join(os.path.dirname(__file__), 'data', 'test_report.csv')
    metric = 'WER'

    with open(test_csv_path, 'rb') as test_csv:
        files = {"file": ("test_results.csv", test_csv, 'text/csv')}
        data = {"metric": metric}
        response = client.post("/process_results", data=data, files=files)

    assert response.status_code == 200

def test_process_csv_cer():
    test_csv_path = os.path.join(os.path.dirname(__file__), 'data', 'test_report.csv')
    metric = 'CER'

    with open(test_csv_path, 'rb') as test_csv:
        files = {"file": ("test_results.csv", test_csv, 'text/csv')}
        data = {"metric": metric}
        response = client.post("/process_results", data=data, files=files)

    assert response.status_code == 200

def test_process_csv_levenshtein_distance():
    test_csv_path = os.path.join(os.path.dirname(__file__), 'data', 'test_report.csv')
    metric = 'Levenshtein Distance'

    with open(test_csv_path, 'rb') as test_csv:
        files = {"file": ("test_results.csv", test_csv, 'text/csv')}
        data = {"metric": metric}
        response = client.post("/process_results", data=data, files=files)

    assert response.status_code == 200


def test_process_csv_levenshtein_distance_with_no_files():
    metric = 'Levenshtein Distance'
    data = {"metric": metric}
    response = client.post("/process_results", data=data)

    assert response.status_code == 400


def test_process_csv_no_metric():
    test_csv_path = os.path.join(os.path.dirname(__file__), 'data', 'test_report.csv')

    with open(test_csv_path, 'rb') as test_csv:
        files = {"file": ("test_results.csv", test_csv, 'text/csv')}
        data = {"metric": []}
        response = client.post("/process_results", data=data, files=files)

    assert response.status_code == 400