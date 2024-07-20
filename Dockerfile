FROM amazon/aws-lambda-python:3.10

RUN rpm -Uvh https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm && \
    yum -y update && \
    yum install -y poppler-utils && \
    yum clean all && \
    rm -rf /var/cache/yum


ENV PIP_DEFAULT_TIMEOUT=600

COPY requirements.txt ${LAMBDA_TASK_ROOT}

RUN pip install -r requirements.txt && \
    pip cache purge

RUN ls -al

COPY lambda_function.py ${LAMBDA_TASK_ROOT}

CMD ["lambda_function.lambda_handler"]
