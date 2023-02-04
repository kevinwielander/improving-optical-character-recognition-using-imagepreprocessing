FROM python:3.9-alpine

WORKDIR /improving-optical-character-recognition-using-imagepreprocessing

COPY requirements.txt .
COPY myenv myenv

RUN pip install -r requirements.txt

COPY . .

RUN . myenv/bin/activate

CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8080"]
