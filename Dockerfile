FROM amazon/aws-lambda-python:3.8

RUN /var/lang/bin/python3.8 -m pip install --upgrade pip

RUN yum install git -y

RUN git clone https://github.com/ottlseo/easy-serverless-rag.git

ENV PIP_DEFAULT_TIMEOUT=600

RUN pip install -r easy-serverless-rag/requirements.txt

RUN pip install "unstructured[all-docs]"

RUN cp easy-serverless-rag/test.py /var/task/

CMD ["test.lambda_handler"]