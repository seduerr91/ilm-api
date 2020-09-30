FROM python:3.7
EXPOSE 8080

ENV APP_HOME /app
WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

COPY . /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]