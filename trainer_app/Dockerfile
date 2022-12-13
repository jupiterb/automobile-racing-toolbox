FROM ubuntu:20.04
FROM python:3.9.6-slim-buster 
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       apt-utils \
       build-essential \
       curl \
       xvfb \
       ffmpeg \
       xorg-dev \
       libsdl2-dev \
       swig \
       cmake \
       python-opengl

ENV PYTHONBUFFERED=1

WORKDIR /app 

COPY ./common /app/common
WORKDIR /app/common 
RUN pip3 install .  

WORKDIR /app 
COPY trainer_app/requirements.txt requirements.txt 
RUN pip3 install --upgrade pip 
RUN pip3 install -r requirements.txt 

COPY ./trainer_app /app
CMD ["python", "-m", "uvicorn",  "--port", "8000", "--host", "0.0.0.0", "src.app:app"]

EXPOSE 8000