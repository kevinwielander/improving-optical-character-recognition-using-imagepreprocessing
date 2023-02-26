import pytesseract


async def read_image(path, lang='deu'):
    try:
        # pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
        pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
        return pytesseract.image_to_string(path, lang=lang)
    except pytesseract.TesseractError as e:
        print("[ERROR] Unable to process image with Tesseract: {0}".format(e))
    except FileNotFoundError as e:
        print("[ERROR] Unable to find file: {0}".format(path))
