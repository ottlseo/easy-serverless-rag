FROM amazon/aws-lambda-python:3.8

RUN /var/lang/bin/python3.8 -m pip install --upgrade pip

RUN yum install git -y

RUN git clone https://github.com/ottlseo/easy-serverless-rag.git

ENV PIP_DEFAULT_TIMEOUT=600

WORKDIR easy-serverless-rag 

RUN pip install -r requirements.txt

RUN pip install "unstructured[all-docs]"

RUN cp lambda_function.py /var/task/

CMD ["lambda_function.lambda_handler"]