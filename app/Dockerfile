# https://hub.docker.com/_/python/
FROM python:3

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

# CMD [ "python", "app.py" ]