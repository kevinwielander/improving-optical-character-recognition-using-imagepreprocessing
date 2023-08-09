# README

## Table of Contents

- [Running application](#running-application)
- [Running the Test Case](#running-the-test-case)
- [Building a Docker Image](#building-a-docker-image)
- [Running a Docker Container](#running-a-docker-container)

---

## Running Application

```bash
.\venv\Scripts\activate
uvicorn main:app --reload       
```


## Running the Test Case


```bash
pytest tests/test_api.py
```
## Building a Docker Image

```bash
docker build -t <docker-image-name> .
```

## Running a Docker Container

```bash
docker run -d -p 8080:8000 <docker-image-name>
```


