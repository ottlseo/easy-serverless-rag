import sys, os
import json
import boto3
from botocore.config import Config

from typing import Optional
from pprint import pprint
# from termcolor import colored
# from utils import bedrock, print_ww
# from utils.bedrock import bedrock_info
from langchain_aws import ChatBedrock
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.embeddings import BedrockEmbeddings
import shutil
from glob import glob
# from utils.common_utils import to_pickle, load_pickle
from unstructured.cleaners.core import clean_bullets, clean_extra_whitespace
from langchain_community.document_loaders import UnstructuredFileLoader, UnstructuredAPIFileLoader

import time
import pickle
import random
import logging
import functools
from IPython.display import Markdown, HTML, display

import cv2
import math
import base64
import numpy as np
from pdf2image import convert_from_path

def to_pickle(obj, path):
    # with open(file=path, mode="wb") as f:
    #     pickle.dump(obj, f)
    # print (f'To_PICKLE: {path}')
    
    # S3에 저장하는 코드로 변경 ---
   
    s3 = boto3.client('s3') 
    pickled_docs = pickle.dumps(obj)
    bucket_name = 'demogo-metadata-source-bucket'
    s3.put_object(
        Bucket=bucket_name,
        Key=path,
        Body=pickled_docs
    )
    print(f"Pickle file uploaded to {bucket_name}/{path}")

    
def load_pickle(path):
    
    s3 = boto3.client('s3') 
    bucket_name = 'demogo-metadata-source-bucket'
    pickle_obj = s3.Object(bucket_name, path)
    pickled_docs = pickle_obj.get()['Body'].read()
    
    obj = pickle.loads(pickled_docs)
    
    # with open(file=path, mode="rb") as f:
    #     obj=pickle.load(f)

    # print (f'Load from {path}')

    return obj

def to_markdown(obj, path):

    with open(file=path, mode="w") as f:
        f.write(obj)

    print (f'To_Markdown: {path}')
    
def print_html(input_html):

    html_string=""
    html_string = html_string + input_html

    display(HTML(html_string))


def get_bedrock_client(
    assumed_role: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    region: Optional[str] = None,
):
    """Create a boto3 client for Amazon Bedrock, with optional configuration overrides

    Parameters
    ----------
    assumed_role :
        Optional ARN of an AWS IAM role to assume for calling the Bedrock service. If not
        specified, the current active credentials will be used.
    endpoint_url :
        Optional override for the Bedrock service API Endpoint. If setting this, it should usually
        include the protocol i.e. "https://..."
    region :
        Optional name of the AWS Region in which the service should be called (e.g. "us-east-1").
        If not specified, AWS_REGION or AWS_DEFAULT_REGION environment variable will be used.
    """
    if region is None:
        target_region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION"))
    else:
        target_region = region

    print(f"Create new client\n  Using region: {target_region}")
    session_kwargs = {"region_name": target_region}
    client_kwargs = {**session_kwargs}

    profile_name = os.environ.get("AWS_PROFILE")
    print(f"  Using profile: {profile_name}")
    if profile_name:
        print(f"  Using profile: {profile_name}")
        session_kwargs["profile_name"] = profile_name

    retry_config = Config(
        region_name=target_region,
        retries={
            "max_attempts": 10,
            "mode": "standard",
        },
    )
    session = boto3.Session(**session_kwargs)

    if assumed_role:
        print(f"  Using role: {assumed_role}", end='')
        sts = session.client("sts")
        response = sts.assume_role(
            RoleArn=str(assumed_role),
            RoleSessionName="langchain-llm-1"
        )
        print(" ... successful!")
        client_kwargs["aws_access_key_id"] = response["Credentials"]["AccessKeyId"]
        client_kwargs["aws_secret_access_key"] = response["Credentials"]["SecretAccessKey"]
        client_kwargs["aws_session_token"] = response["Credentials"]["SessionToken"]

    if endpoint_url:
        client_kwargs["endpoint_url"] = endpoint_url

    bedrock_client = session.client(
        service_name="bedrock-runtime",
        config=retry_config,
        **client_kwargs
    )

    print("boto3 Bedrock client successfully created!")
    print(bedrock_client._endpoint)
    return bedrock_client


class bedrock_info():

    _BEDROCK_MODEL_INFO = {
        "Claude-Instant-V1": "anthropic.claude-instant-v1",
        "Claude-V1": "anthropic.claude-v1",
        "Claude-V2": "anthropic.claude-v2",
        "Claude-V2-1": "anthropic.claude-v2:1",
        "Claude-V3-Sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
        "Claude-V3-Haiku": "anthropic.claude-3-haiku-20240307-v1:0",
        "Jurassic-2-Mid": "ai21.j2-mid-v1",
        "Jurassic-2-Ultra": "ai21.j2-ultra-v1",
        "Command": "cohere.command-text-v14",
        "Command-Light": "cohere.command-light-text-v14",
        "Cohere-Embeddings-En": "cohere.embed-english-v3",
        "Cohere-Embeddings-Multilingual": "cohere.embed-multilingual-v3",
        "Titan-Embeddings-G1": "amazon.titan-embed-text-v1",
        "Titan-Text-Embeddings-V2": "amazon.titan-embed-text-v2:0",
        "Titan-Text-G1": "amazon.titan-text-express-v1",
        "Titan-Text-G1-Light": "amazon.titan-text-lite-v1",
        "Titan-Text-G1-Premier": "amazon.titan-text-premier-v1:0",
        "Titan-Text-G1-Express": "amazon.titan-text-express-v1",
        "Llama2-13b-Chat": "meta.llama2-13b-chat-v1"
    }

    @classmethod
    def get_list_fm_models(cls, verbose=False):

        if verbose:
            bedrock = boto3.client(service_name='bedrock')
            model_list = bedrock.list_foundation_models()
            return model_list["modelSummaries"]
        else:
            return cls._BEDROCK_MODEL_INFO

    @classmethod
    def get_model_id(cls, model_name):

        assert model_name in cls._BEDROCK_MODEL_INFO.keys(), "Check model name"

        return cls._BEDROCK_MODEL_INFO[model_name]


def add_python_path(module_path):
    if os.path.abspath(module_path) not in sys.path:
        sys.path.append(os.path.abspath(module_path))
        print(f"python path: {os.path.abspath(module_path)} is added")
    else:
        print(f"python path: {os.path.abspath(module_path)} already exists")
    print("sys.path: ", sys.path)

module_path = "../../.."
add_python_path(module_path)

# Step 1. Create Bedrock client
boto3_bedrock = get_bedrock_client(
    assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    endpoint_url=os.environ.get("BEDROCK_ENDPOINT_URL", None),
    region=os.environ.get("AWS_DEFAULT_REGION", None),
)

pprint (bedrock_info.get_list_fm_models(verbose=False))

# Step 2. Titan Embedding 및 LLM 인 Claude-v3-sonnet 모델 로딩
llm_text = ChatBedrock(
    model_id=bedrock_info.get_model_id(model_name="Claude-V3-Sonnet"),
    client=boto3_bedrock,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    model_kwargs={
        "max_tokens": 2048,
        "stop_sequences": ["\n\nHuman"],
        # "temperature": 0,
        # "top_k": 350,
        # "top_p": 0.999
    }
)
# Embedding model 선택
llm_emb = BedrockEmbeddings(
    client=boto3_bedrock,
    model_id=bedrock_info.get_model_id(model_name="Titan-Embeddings-G1") #Titan-Text-Embeddings-V2
)
dimension = 1536 #1024
print("Bedrock Embeddings Model Loaded")

# 3. 데이터 준비
image_path = "./fig"
file_path = "./data/complex_pdf/school_edu_guide.pdf"

if os.path.isdir(image_path): shutil.rmtree(image_path)
os.mkdir(image_path)

loader = UnstructuredFileLoader(
    file_path=file_path,

    chunking_strategy = "by_title",
    mode="elements",

    strategy="hi_res",
    hi_res_model_name="yolox", #"detectron2_onnx", "yolox", "yolox_quantized"

    extract_images_in_pdf=True,
    #skip_infer_table_types='[]', # ['pdf', 'jpg', 'png', 'xls', 'xlsx', 'heic']
    pdf_infer_table_structure=True, ## enable to get table as html using tabletrasformer

    extract_image_block_output_dir=image_path,
    extract_image_block_to_payload=False, ## False: to save image

    max_characters=4096,
    new_after_n_chars=4000,
    combine_text_under_n_chars=2000,

    languages= ["kor+eng"],

    post_processors=[clean_bullets, clean_extra_whitespace]
)
docs = loader.load()

to_pickle(docs, "./data/complex_pdf/pickle/parsed_unstructured.pkl")
docs = load_pickle("./data/complex_pdf/pickle/parsed_unstructured.pkl")

tables, texts = [], []
images = glob(os.path.join(image_path, "*"))

tables, texts = [], []

for doc in docs:

    category = doc.metadata["category"]

    if category == "Table": tables.append(doc)
    elif category == "Image": images.append(doc)
    else: texts.append(doc)
    
    images = glob(os.path.join(image_path, "*"))

print (f' # texts: {len(texts)} \n # tables: {len(tables)} \n # images: {len(images)}')

def image_to_base64(image_path):
    
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        
    return encoded_string.decode('utf-8')

table_as_image = True

if table_as_image:
    image_tmp_path = os.path.join(image_path, "tmp")
    if os.path.isdir(image_tmp_path): shutil.rmtree(image_tmp_path)
    os.mkdir(image_tmp_path)
    
    # from pdf to image
    pages = convert_from_path(file_path)
    for i, page in enumerate(pages):
        print (f'pdf page {i}, size: {page.size}')    
        page.save(f'{image_tmp_path}/{str(i+1)}.jpg', "JPEG")

    print ("==")

    #table_images = []
    for idx, table in enumerate(tables):
        points = table.metadata["coordinates"]["points"]
        page_number = table.metadata["page_number"]
        layout_width, layout_height = table.metadata["coordinates"]["layout_width"], table.metadata["coordinates"]["layout_height"]

        img = cv2.imread(f'{image_tmp_path}/{page_number}.jpg')
        crop_img = img[math.ceil(points[0][1]):math.ceil(points[1][1]), \
                       math.ceil(points[0][0]):math.ceil(points[3][0])]
        table_image_path = f'{image_path}/table-{idx}.jpg'
        cv2.imwrite(table_image_path, crop_img)
        #table_images.append(table_image_path)

        print (f'unstructured width: {layout_width}, height: {layout_height}')
        print (f'page_number: {page_number}')
        print ("==")

        width, height, _ = crop_img.shape
        image_token = width*height/750
        print (f'image: {table_image_path}, shape: {img.shape}, image_token_for_claude3: {image_token}' )

        ## Resize image
        if image_token > 1500:
            resize_img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            print("   - resize_img.shape = {0}".format(resize_img.shape))
            table_image_resize_path = table_image_path.replace(".jpg", "-resize.jpg")
            cv2.imwrite(table_image_resize_path, resize_img)
            os.remove(table_image_path)
            table_image_path = table_image_resize_path

        img_base64 = image_to_base64(table_image_path)
        table.metadata["image_base64"] = img_base64

    if os.path.isdir(image_tmp_path): shutil.rmtree(image_tmp_path)
    #print (f'table_images: {table_images}')
    images = glob(os.path.join(image_path, "*"))
    print (f'images: {images}')
