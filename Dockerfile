FROM amazon/aws-lambda-python:3.10

RUN /var/lang/bin/python3.10 -m pip install --upgrade pip

RUN yum install git -y

RUN git clone https://github.com/ottlseo/multimodal-rag-made-easy.git

ENV PIP_DEFAULT_TIMEOUT=600

WORKDIR multimodal-rag-made-easy

RUN rpm -Uvh https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm

RUN yum -y update

RUN yum install -y poppler-utils

RUN pip install -r requirements.txt

RUN pip install -U "unstructured[all-docs]==0.13.2"

RUN cp lambda_function.py /var/task/

CMD ["lambda_function.lambda_handler"]
