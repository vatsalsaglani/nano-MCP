FROM python:3.12

WORKDIR /app

RUN apt-get update && apt-get install -y git

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /data

CMD ["python", "server.py"]

