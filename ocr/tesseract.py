import pytesseract


# TODO: replace lang string with environment variable
async def read_image(path, lang='eng'):
    try:
        pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
        return pytesseract.image_to_string(path, lang=lang)
    except pytesseract.TesseractError as e:
        print("[ERROR] Unable to process image with Tesseract: {0}".format(e))
    except FileNotFoundError as e:
        print("[ERROR] Unable to find file: {0}".format(path))


def test():
    return True
