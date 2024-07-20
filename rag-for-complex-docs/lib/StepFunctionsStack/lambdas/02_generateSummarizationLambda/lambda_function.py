import time, os, sys
import base64
import boto3
from botocore.config import Config
from typing import Optional
from langchain.schema import Document
from langchain_core.messages import HumanMessage
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

#### ======== FUNCTIONS ======== ####
def get_bedrock_client(
    assumed_role: Optional[str] = None, # Optional - If not specified, the current active credentials will be used
    endpoint_url: Optional[str] = None, # Optional - If setting this, it should usually include the protocol (i.e. "https://...")
    region: Optional[str] = None, # Optional - If not specified, AWS_REGION or AWS_DEFAULT_REGION environment variable will be used
):
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

def load_llms(boto3_bedrock):
    llm = ChatBedrock(
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
    llm_emb = BedrockEmbeddings(
        client=boto3_bedrock,
        model_id=bedrock_info.get_model_id(model_name="Titan-Embeddings-G1")
    )
    dimension = 1536 # 1024
    return llm, llm_emb, dimension

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return encoded_string.decode('utf-8')


#### ======== HANDLER ======== ####


boto3_bedrock = get_bedrock_client(
    assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    endpoint_url=os.environ.get("BEDROCK_ENDPOINT_URL", None),
    region=os.environ.get("AWS_DEFAULT_REGION", None),
)
llm_text, llm_emb, dimension = load_llms(boto3_bedrock)




system_prompt = "You are an assistant tasked with describing table and image."
system_message_template = SystemMessagePromptTemplate.from_template(system_prompt)

## For images
human_prompt = [
    {
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64," + "{image_base64}",
        },
    },
    {
        "type": "text",
        "text": '''
                 Given image, give a concise summary.
                 Don't insert any XML tag such as <text> and </text> when answering.
                 Write in Korean.
        '''
    },
]

human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)

prompt = ChatPromptTemplate.from_messages(
    [
        system_message_template,
        human_message_template
    ]
)

summarize_chain = prompt | llm_text | StrOutputParser()
#summarize_chain = {"image_base64": lambda x:x} | prompt | llm_text | StrOutputParser()

img_info = [image_to_base64(img_path) for img_path in images if os.path.basename(img_path).startswith("figure")]

def summary_img(summarize_chain, img_base64):

    img = Image.open(BytesIO(base64.b64decode(img_base64)))
    plt.imshow(img)
    plt.show()

    summary = summarize_chain.invoke(
        {
            "image_base64": img_base64
        }
    )

    return summary

image_summaries = []
for idx, img_base64 in enumerate(img_info):
    summary = summary_img(summarize_chain, img_base64)
    image_summaries.append(summary)
    # print ("\n==")
    # print (idx)
    
#image_summaries = summarize_chain.batch(img_info, config={"max_concurrency": 1})


verbose = True
if verbose:
    for img_base64, summary in zip(img_info, image_summaries):

        print ("============================")
        img = Image.open(BytesIO(base64.b64decode(img_base64)))
        plt.imshow(img)
        plt.show()
