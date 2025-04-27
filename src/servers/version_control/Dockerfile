FROM python:3.12

RUN apt-get update && apt-get install -y git

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /repo

# Update the git config to trust all directories in /repo recursively
RUN git config --global --add safe.directory /repo
RUN git config --global --add safe.directory '*'

# Set default Git user config (required for commits)
RUN git config --global user.email "docker@example.com"
RUN git config --global user.name "Docker User"

CMD ["python", "server.py"]

