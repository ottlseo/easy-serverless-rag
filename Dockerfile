FROM amazon/aws-lambda-python:3.8

RUN /var/lang/bin/python3.8 -m pip install --upgrade pip

RUN yum install git -y

RUN git clone https://github.com/ottlseo/easy-serverless-rag.git

RUN pip install -r easy-serverless-rag/requirements.txt
RUN pip install "unstructured[all-docs]"

RUN cp easy-serverless-rag/lambda_function.py /var/task/

CMD ["lambda_function.handler"]