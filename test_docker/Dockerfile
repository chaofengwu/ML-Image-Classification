From python:3.6.3

WORKDIR /app

ADD . /app

RUN pip3 install --trusted-host pypi.python.org -r requirements.txt

EXPOSE 80

ENV NAME World

CMD ["python3", "app.py"]
