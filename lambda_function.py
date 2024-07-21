import json
import os, sys
import boto3
import botocore
# from botocore.config import Config
from typing import Optional
import shutil
from glob import glob
import pickle
import cv2
import math
import base64
# import numpy as np
from langchain_community.chat_models import BedrockChat
from langchain_aws import ChatBedrock
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import BedrockEmbeddings
from langchain_community.document_loaders import UnstructuredURLLoader, UnstructuredFileLoader, S3FileLoader
from unstructured.cleaners.core import clean_bullets, clean_extra_whitespace
from pdf2image import convert_from_path
import nltk
# from huggingface_hub import hf_hub_download

# S3 클라이언트 생성
s3 = boto3.client('s3')

bucket_name = 'demogo-metadata-source-bucket'
image_path = "/tmp/fig" # "./fig"
os.makedirs(image_path, exist_ok=True)

nltk.data.path.append("/tmp/nltk_data")
# if not os.path.exists("/tmp/nltk_data/punkt"):
nltk.download("punkt", download_dir="/tmp/nltk_data")
nltk.download("averaged_perceptron_tagger", download_dir="/tmp/nltk_data")

# model_name = "yolox"
# hf_hub_download(filename=model_name, repo_id="julien-c/EsperBERTo-small", cache_dir="/tmp") # invalid

table_by_llama_parse = False
table_by_pymupdf = False
table_as_image = True

# =============== FUNCTIONS =============== #
# def add_python_path(module_path):
#     if os.path.abspath(module_path) not in sys.path:
#         sys.path.append(os.path.abspath(module_path))
#         print(f"python path: {os.path.abspath(module_path)} is added")
#     else:
#         print(f"python path: {os.path.abspath(module_path)} already exists")
#     print("sys.path: ", sys.path)

# module_path = "../../.."
# add_python_path(module_path)

def check_if_s3_file_exist(filename):
    try:
        s3.head_object(Bucket=bucket_name, Key=filename)
        return True
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        else:
            raise e

def get_file_url_from_s3(file_key):
    s3url = s3.generate_presigned_url(
        ClientMethod='get_object',
        Params={'Bucket': bucket_name, 'Key': file_key},
        ExpiresIn=3600  # expired time
    )
    return s3url

def get_file_from_s3_to_local(file_key):
    response = s3.get_object(Bucket=bucket_name, Key=file_key)
    file_content = response['Body'].read()
    local_file_path = f'/tmp/{file_key}'
    with open(local_file_path, 'wb') as f:
        f.write(file_content)
    return local_file_path

def to_pickle(obj, path):
    with open(file=path, mode="wb") as f:
        pickle.dump(obj, f)
    print (f'To_PICKLE: {path}')

def to_pickle_to_s3(obj, path):   
    pickled_docs = pickle.dumps(obj)
    s3.put_object(
        Bucket=bucket_name,
        Key=path,
        Body=pickled_docs
    )
    print(f"Pickle file uploaded to {bucket_name}/{path}")
    
def load_pickle(path):
    with open(file=path, mode="rb") as f:
        obj=pickle.load(f)
    print (f'Load from {path}')

def load_pickle_from_s3(path):
    pickle_obj = s3.get_object(Bucket=bucket_name, Key=path)
    pickled_docs = pickle_obj['Body'].read()
    obj = pickle.loads(pickled_docs)
    return obj

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return encoded_string.decode('utf-8')


# =============== HANDLER =============== #
def lambda_handler(event, context):
    
    if check_if_s3_file_exist(event['filename']): #'school_edu_guide.pdf'
        file_key = event['filename']
    else:
        raise FileNotFoundError(f"file not found.")
    
    if os.path.isdir(image_path): shutil.rmtree(image_path)
    os.mkdir(image_path)
    
    # option 1: S3 URL 가져와 URLLoader 사용하기
    # file_path = get_file_path_from_s3()
    # loader = UnstructuredURLLoader(urls=[file_path],
    
    # option 2: S3 파일을 로컬로 가져와 FileLoader 사용하기
    file_path = get_file_from_s3_to_local(file_key)
    loader = UnstructuredFileLoader(file_path=file_path,
    
    # option 3: S3FileLoader 사용하기
    # loader = S3FileLoader(bucket=bucket_name, key=file_key,
        chunking_strategy = "by_title",
        mode="elements",
        
        strategy="fast", # 레이아웃 자동 검사 - 모델 사용(hi_res) 또는 미사용(fast)
        # strategy="hi_res", 
        # hi_res_model_name="yolox", # hi_res 선택 시, 사용될 모델 #"detectron2_onnx", "yolox", "yolox_quantized"
    
        extract_images_in_pdf=True, # 이미지 추출 여부
        #skip_infer_table_types='[]', # ['pdf', 'jpg', 'png', 'xls', 'xlsx', 'heic']
        pdf_infer_table_structure=True, ## enable to get table as html using tabletrasformer
    
        extract_image_block_to_payload=False, # 이미지를 파일로 저장할지(False) base64로 인코딩해 메타데이터화할지(True)
        extract_image_block_output_dir=image_path, # 이미지 파일 저장 위치
    
        max_characters=4096,
        new_after_n_chars=4000,
        combine_text_under_n_chars=2000,
    
        languages= ["kor+eng"],
    
        post_processors=[clean_bullets, clean_extra_whitespace]
    )
    docs = loader.load()
    
    test_file_key = 'parsed_unstructured.txt'
    s3.put_object(
            Bucket=bucket_name,
            Key=test_file_key,
            Body=str(docs)
    )

    new_file_key = 'parsed_unstructured.pkl'
    to_pickle_to_s3(docs, new_file_key)   
    docs = load_pickle_from_s3(new_file_key)

    # new_file_key = 'tmp/parsed_unstructured.pkl'
    # to_pickle(docs, new_file_key)   
    # docs = load_pickle(new_file_key)

    tables, texts = [], []
    images = glob(os.path.join(image_path, "*"))

    for doc in docs:
        category = doc.metadata["category"]
        if category == "Table": tables.append(doc)
        elif category == "Image": images.append(doc)
        else: texts.append(doc) 
        images = glob(os.path.join(image_path, "*"))

    print (f' # texts: {len(texts)} \n # tables: {len(tables)} \n # images: {len(images)}') 

    # if table_as_image:
    #     image_tmp_path = os.path.join(image_path, "tmp")
    #     if os.path.isdir(image_tmp_path): shutil.rmtree(image_tmp_path)
    #     os.mkdir(image_tmp_path)
        
    #     # from pdf to image
    #     pages = convert_from_path(file_path)
    #     for i, page in enumerate(pages):
    #         print (f'pdf page {i}, size: {page.size}')    
    #         page.save(f'{image_tmp_path}/{str(i+1)}.jpg', "JPEG")

    #     print ("==")

    #     #table_images = []
    #     for idx, table in enumerate(tables):
    #         points = table.metadata["coordinates"]["points"]
    #         page_number = table.metadata["page_number"]
    #         layout_width, layout_height = table.metadata["coordinates"]["layout_width"], table.metadata["coordinates"]["layout_height"]

    #         img = cv2.imread(f'{image_tmp_path}/{page_number}.jpg')
    #         crop_img = img[math.ceil(points[0][1]):math.ceil(points[1][1]), \
    #                     math.ceil(points[0][0]):math.ceil(points[3][0])]
    #         table_image_path = f'{image_path}/table-{idx}.jpg'
    #         cv2.imwrite(table_image_path, crop_img)
    #         #table_images.append(table_image_path)

    #         print (f'unstructured width: {layout_width}, height: {layout_height}')
    #         print (f'page_number: {page_number}')
    #         print ("==")

    #         width, height, _ = crop_img.shape
    #         image_token = width*height/750
    #         print (f'image: {table_image_path}, shape: {img.shape}, image_token_for_claude3: {image_token}' )

    #         ## Resize image
    #         if image_token > 1500:
    #             resize_img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    #             print("   - resize_img.shape = {0}".format(resize_img.shape))
    #             table_image_resize_path = table_image_path.replace(".jpg", "-resize.jpg")
    #             cv2.imwrite(table_image_resize_path, resize_img)
    #             os.remove(table_image_path)
    #             table_image_path = table_image_resize_path

    #         img_base64 = image_to_base64(table_image_path)
    #         table.metadata["image_base64"] = img_base64

    #     if os.path.isdir(image_tmp_path): shutil.rmtree(image_tmp_path)
    #     #print (f'table_images: {table_images}')
    #     images = glob(os.path.join(image_path, "*"))
    #     print (f'images: {images}')


# =============== FOR TEST =============== #
    prompt = event['prompt']
    print("\n\n\nLOADING PDF IS DONE!!\n")

    return {
        'statusCode': 200,
        'body': json.dumps(f' Hi, Your prompt was ... {prompt}')
    }
