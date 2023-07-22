import pytesseract
import logging
import platform
logger = logging.getLogger(__name__)


async def read_image(path, lang='deu'):
    try:
        # Detect the operating system and set the tesseract_cmd accordingly
        if platform.system() == 'Windows':
            pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
        elif platform.system() == 'Linux':
            pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
        else:
            raise OSError('Unsupported operating system')

        logger.info('Reading Image using Tesseract')
        return pytesseract.image_to_string(path, lang=lang)
    except Exception as e:
        logger.error('Failed to read image: %s', str(e))
        raise e
