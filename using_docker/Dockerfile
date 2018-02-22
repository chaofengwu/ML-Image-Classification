From python 3.6.3

WORKDIR /test

ADD . /test

RUN pip3 install --trusted-host pypi.python.org -r requirements.txt

EXPOSE 80

ENV NAME World

CMD ["python3", "main.py", "test", "", "1", ""]