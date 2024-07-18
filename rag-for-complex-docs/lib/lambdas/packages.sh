#!/bin/bash

set -e


sudo -u ec2-user -i <<'EOF'
 
#source /home/ec2-user/anaconda3/bin/deactivate
pip==24.1.1
awscli==1.33.16
botocore==1.34.134
boto3==1.34.134
sagemaker==2.224.1
langchain==0.2.6
langchain-community==0.2.6
langchain_aws==0.1.8
termcolor==2.4.0
transformers==4.41.2
librosa==0.10.2.post1
opensearch-py==2.6.0
sqlalchemy #==2.0.1
pypdf==4.2.0
#spacy
# spacy download ko_core_news_md
ipython==8.25.0
ipywidgets==8.1.3
#llmsherpa
anthropic==0.30.0
faiss-cpu==1.8.0.post1
jq==1.7.0
pydantic==2.7.4

sudo rpm -Uvh https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm
sudo yum -y update
sudo yum install -y poppler-utils
lxml==5.2.2
kaleido==0.2.1
uvicorn==0.30.1
pandas==2.2.2
numexpr==2.10.1
pdf2image==1.17.0

#sudo sh install_tesseract.sh
#sudo sh SageMaker/aws-ai-ml-workshop-kr/genai/aws-gen-ai-kr/00_setup/install_tesseract.sh
sudo amazon-linux-extras install libreoffice -y
"unstructured[all-docs]==0.13.2"
#sudo rm -rf leptonica-1.84.1 leptonica-1.84.1.tar.gz tesseract-ocr

python-dotenv==1.0.1
llama-parse==0.4.4
pymupdf==1.24.7

EOF