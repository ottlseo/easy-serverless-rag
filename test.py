import json
import boto3
from langchain_community.chat_models import BedrockChat
# from langchain_aws import ChatBedrock
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# from langchain.embeddings import BedrockEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader, UnstructuredURLLoader
from unstructured.cleaners.core import clean_bullets, clean_extra_whitespace

# S3 클라이언트 생성
s3 = boto3.client('s3')

# 버킷 이름과 파일 키 설정
bucket_name = 'demogo-metadata-source-bucket'
file_key = 'school_edu_guide.pdf'
image_path = "./fig"

def lambda_handler(event, context):
    llm = BedrockChat( #BedrockChat llm 클라이언트 생성
        model_id="anthropic.claude-3-sonnet-20240229-v1:0", # Claude 3 Sonnet 모델 선택
        model_kwargs={
            "max_tokens": 512,
            "temperature": 0
        }
    )
    prompt = event['prompt']
    response_text = llm.invoke(prompt) #프롬프트에 응답 반환
    
    print(response_text.content)
    
    # try:
    #     response = s3.get_object(Bucket=bucket_name, Key=file_key)
    #     file_content = response['Body'].read().decode('utf-8')
    #     print(file_content)
    # except Exception as e:
    #     print(f'Error downloading file: {e}')
    
    # S3 파일에 대한 URL 생성
    url = s3.generate_presigned_url(
        ClientMethod='get_object',
        Params={'Bucket': bucket_name, 'Key': file_key},
        ExpiresIn=3600  # expired time
    )

    # loader = UnstructuredFileLoader(
    #     file_path=file_path,
    
    loader = UnstructuredURLLoader(
        urls=[url],
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
    
    print("\n\n\nDONE\n")

    return {
        'statusCode': 200,
        'body': json.dumps('Index setting done from Lambda!')
    }
