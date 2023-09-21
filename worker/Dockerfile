FROM python:3.10
WORKDIR /app/worker
COPY . .
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --no-cache-dir -r requirements.txt
CMD [ "python", "./main.py" ]