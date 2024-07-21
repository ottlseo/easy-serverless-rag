# Multimodal RAG Made Easy

This project proposes an architecture and demo of building a multimodal RAG on AWS as a serverless and CDK. 

The code of the two projects below shows how to build a multimodal RAG. 
- **aws-bedrock-examples Github**: [multimodal-rag-pdf.ipynb](https://github.com/aws-samples/amazon-bedrock-samples/blob/main/rag-solutions/multimodal-rag-pdf/rag/multimodal-rag-pdf.ipynb)
- **aws-ai-ml-workshop-kr Github**: [05_0_load_complex_pdf_kr_opensearch.ipynb](https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/genai/aws-gen-ai-kr/20_applications/02_qa_chatbot/01_preprocess_docs/05_0_load_complex_pdf_kr_opensearch.ipynb)

## Architecture
To make the multimodal RAG implemented in the ipynb environment above available to the application by calling them as events, this project migrate them to a serverless environment. To do this, we break the code into multiple Lambdas per module and orchestrate them using AWS Step Functions. 

The Step Functions workflow builds a multimodal RAG in 3 steps.

1. Load unstructed files using [UnstructedFileLoader](https://python.langchain.com/v0.2/docs/integrations/document_loaders/unstructured_file/) (or [S3FileLoader](https://api.python.langchain.com/en/latest/_modules/langchain_community/document_loaders/s3_file.html#S3FileLoader))
2. Generate summarization of Images or Tables using [Step Functions Map state](https://docs.aws.amazon.com/ko_kr/step-functions/latest/dg/amazon-states-language-map-state.html) to prevent Lambda Timeout issue (Summarized by Anthropic Claude Sonnet 3.0)
3. Start vector embedding using [Amazon Titan Text Embeddings V2 model](https://docs.aws.amazon.com/ko_kr/bedrock/latest/userguide/titan-embedding-models.html) and indexing using Amazon OpenSearch Serverless

![demogo-ottlseo-0720-advanced-multimodal-rag drawio](https://github.com/user-attachments/assets/bb516990-d1a9-4b83-9197-903dca3c2ec0)

## Issues
TBD

## Contributor
- Yoonseo Kim, AWS Associate Solutions Architect   
- TBD
