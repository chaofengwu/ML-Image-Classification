FROM python:3.6.3

WORKDIR /test

ADD . /test

RUN pip3 install --trusted-host pypi.python.org -r requirements.txt

EXPOSE 80

ENV NAME World

CMD ["python3", "main.py", "--mode", "test", "--folder_path", "/test/images/", "--folder_mode", "1", "--label", "label_no_wrong_file.txt"]
