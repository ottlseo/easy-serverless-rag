FROM amazon/aws-lambda-python:3.10

RUN rpm -Uvh https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm

RUN yum -y update

RUN yum install git poppler-utils -y

RUN /var/lang/bin/python3.10 -m pip install --upgrade pip

ENV PIP_DEFAULT_TIMEOUT=600

COPY requirements.txt ${LAMBDA_TASK_ROOT}

RUN pip install -r requirements.txt

RUN pip install -U "unstructured[pdf]==0.13.2"

RUN ls -al

COPY lambda_function.py ${LAMBDA_TASK_ROOT}

CMD ["lambda_function.lambda_handler"]
