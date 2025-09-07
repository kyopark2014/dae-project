import traceback
import boto3
import os
import json
import re
import uuid
import base64
import info 
import PyPDF2
import csv
import utils
import strands_agent
import langgraph_agent
import mcp_config

from io import BytesIO
from PIL import Image
from langchain_aws import ChatBedrock
from botocore.config import Config
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.docstore.document import Document
from tavily import TavilyClient  
from urllib import parse
from pydantic.v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, AIMessageChunk
from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from multiprocessing import Process, Pipe

import logging
import sys

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)

logger = logging.getLogger("chat")

mcp_config.initialize_config()

reasoning_mode = 'Disable'
debug_messages = []  # List to store debug messages

config = utils.load_config()
print(f"config: {config}")

region = config.get("region", "us-west-2")
projectName = config.get("projectName", "dae")

accountId = config.get("accountId", None)
if accountId is None:
    raise Exception ("No accountId")

s3_prefix = 'docs'
s3_image_prefix = 'images'

MSG_LENGTH = 100    

doc_prefix = s3_prefix+'/'

model_name = "Claude 3.5 Sonnet"
model_type = "claude"
models = info.get_model_info(model_name)
number_of_models = len(models)
model_id = models[0]["model_id"]
debug_mode = "Enable"
multi_region = "Disable"

aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
aws_session_token = os.environ.get('AWS_SESSION_TOKEN')
aws_region = os.environ.get('AWS_DEFAULT_REGION', 'us-west-2')

reasoning_mode = 'Disable'
grading_mode = 'Disable'
agent_type = 'langgraph'
user_id = agent_type # for testing

def update(modelName, debugMode, multiRegion, reasoningMode, gradingMode, agentType):    
    global model_name, model_id, model_type, debug_mode, multi_region, reasoning_mode, grading_mode
    global models, user_id, agent_type

    # load mcp.env    
    mcp_env = utils.load_mcp_env()
    
    if model_name != modelName:
        model_name = modelName
        logger.info(f"model_name: {model_name}")
        
        models = info.get_model_info(model_name)
        model_id = models[0]["model_id"]
        model_type = models[0]["model_type"]
                                
    if debug_mode != debugMode:
        debug_mode = debugMode        
        logger.info(f"debug_mode: {debug_mode}")

    if reasoning_mode != reasoningMode:
        reasoning_mode = reasoningMode
        logger.info(f"reasoning_mode: {reasoning_mode}")    

    if multi_region != multiRegion:
        multi_region = multiRegion
        logger.info(f"multi_region: {multi_region}")
        mcp_env['multi_region'] = multi_region

    if grading_mode != gradingMode:
        grading_mode = gradingMode
        logger.info(f"grading_mode: {grading_mode}")            
        mcp_env['grading_mode'] = grading_mode

    if agent_type != agentType:
        agent_type = agentType
        logger.info(f"agent_type: {agent_type}")

        logger.info(f"agent_type changed, update memory variables.")
        user_id = agent_type
        logger.info(f"user_id: {user_id}")
        mcp_env['user_id'] = user_id

    # update mcp.env    
    utils.save_mcp_env(mcp_env)
    # logger.info(f"mcp.env updated: {mcp_env}")

def update_mcp_env():
    mcp_env = utils.load_mcp_env()
    
    mcp_env['multi_region'] = multi_region
    mcp_env['grading_mode'] = grading_mode
    user_id = agent_type
    mcp_env['user_id'] = user_id

    utils.save_mcp_env(mcp_env)
    logger.info(f"mcp.env updated: {mcp_env}")

map_chain = dict() 
checkpointers = dict() 
memorystores = dict() 

checkpointer = MemorySaver()
memorystore = InMemoryStore()

def initiate():
    global memory_chain, checkpointer, memorystore, checkpointers, memorystores

    if user_id in map_chain:  
        logger.info(f"memory exist. reuse it!")
        memory_chain = map_chain[user_id]

        checkpointer = checkpointers[user_id]
        memorystore = memorystores[user_id]
    else: 
        logger.info(f"memory not exist. create new memory!")
        memory_chain = ConversationBufferWindowMemory(memory_key="chat_history", output_key='answer', return_messages=True, k=5)
        map_chain[user_id] = memory_chain

        checkpointer = MemorySaver()
        memorystore = InMemoryStore()

        checkpointers[user_id] = checkpointer
        memorystores[user_id] = memorystore

def clear_chat_history():
    # Initialize memory chain
    initiate()
    
    global memory_chain
    memory_chain = []
    map_chain[user_id] = memory_chain

def save_chat_history(text, msg):
    # Initialize memory chain
    initiate()
    
    memory_chain.chat_memory.add_user_message(text)
    if len(msg) > MSG_LENGTH:
        memory_chain.chat_memory.add_ai_message(msg[:MSG_LENGTH])                          
    else:
        memory_chain.chat_memory.add_ai_message(msg) 

def create_object(key, body):
    """
    Create an object in S3 and return the URL. If the file already exists, append the new content.
    """
    
    # Content-Type based on file extension
    content_type = 'application/octet-stream'  # default value
    if key.endswith('.html'):
        content_type = 'text/html'
    elif key.endswith('.md'):
        content_type = 'text/markdown'
    
    if aws_access_key and aws_secret_key:
        s3_client = boto3.client(
            service_name='s3',
            region_name=region,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            aws_session_token=aws_session_token,
        )
    else:
        s3_client = boto3.client(
            service_name='s3',
            region_name=region,
        )
        
    s3_client.put_object(
        Bucket=s3_bucket,
        Key=key,
        Body=body,
        ContentType=content_type
    )  

def updata_object(key, body, direction):
    """
    Create an object in S3 and return the URL. If the file already exists, append the new content.
    """
    if aws_access_key and aws_secret_key:
        s3_client = boto3.client(
            service_name='s3',
            region_name=region,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            aws_session_token=aws_session_token,
        )
    else:
        s3_client = boto3.client(
            service_name='s3',
            region_name=region,
        )

    try:
        # Check if file exists
        try:
            response = s3_client.get_object(Bucket=s3_bucket, Key=key)
            existing_body = response['Body'].read().decode('utf-8')
            # Append new content to existing content

            if direction == 'append':
                updated_body = existing_body + '\n' + body
            else: # prepend
                updated_body = body + '\n' + existing_body
        except s3_client.exceptions.NoSuchKey:
            # File doesn't exist, use new body as is
            updated_body = body
            
        # Content-Type based on file extension
        content_type = 'application/octet-stream'  # default value
        if key.endswith('.html'):
            content_type = 'text/html'
        elif key.endswith('.md'):
            content_type = 'text/markdown'
            
        # Upload the updated content
        s3_client.put_object(
            Bucket=s3_bucket,
            Key=key,
            Body=updated_body,
            ContentType=content_type
        )
        
    except Exception as e:
        logger.error(f"Error updating object in S3: {str(e)}")
        raise e

selected_chat = 0
def get_chat(extended_thinking):
    global selected_chat, model_type

    logger.info(f"models: {models}")
    logger.info(f"selected_chat: {selected_chat}")
    
    profile = models[selected_chat]
    # print('profile: ', profile)
        
    region =  profile['bedrock_region']
    modelId = profile['model_id']
    model_type = profile['model_type']
    if model_type == 'claude':
        maxOutputTokens = 4096 # 4k
    else:
        maxOutputTokens = 5120 # 5k
    number_of_models = len(models)

    logger.info(f"LLM: {selected_chat}, region: {region}, modelId: {modelId}, model_type: {model_type}")

    if profile['model_type'] == 'nova':
        STOP_SEQUENCE = '"\n\n<thinking>", "\n<thinking>", " <thinking>"'
    elif profile['model_type'] == 'claude':
        STOP_SEQUENCE = "\n\nHuman:" 
    elif profile['model_type'] == 'openai':
        STOP_SEQUENCE = "" 
                          
    # bedrock   
    if aws_access_key and aws_secret_key:
        boto3_bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name=region,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            aws_session_token=aws_session_token,
            config=Config(
                retries = {
                    'max_attempts': 30
                }
            )
        )
    else:
        boto3_bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name=region,
            config=Config(
                retries = {
                    'max_attempts': 30
                }
            )
        )

    if profile['model_type'] != 'openai' and extended_thinking=='Enable':
        maxReasoningOutputTokens=64000
        logger.info(f"extended_thinking: {extended_thinking}")
        thinking_budget = min(maxOutputTokens, maxReasoningOutputTokens-1000)

        parameters = {
            "max_tokens":maxReasoningOutputTokens,
            "temperature":1,            
            "thinking": {
                "type": "enabled",
                "budget_tokens": thinking_budget
            },
            "stop_sequences": [STOP_SEQUENCE]
        }
    elif profile['model_type'] != 'openai' and extended_thinking=='Disable':
        parameters = {
            "max_tokens":maxOutputTokens,     
            "temperature":0.1,
            "top_k":250,
            "top_p":0.9,
            "stop_sequences": [STOP_SEQUENCE]
        }
    elif profile['model_type'] == 'openai':
        parameters = {
            "max_tokens":maxOutputTokens,     
            "temperature":0.1,
            "top_k":250,
            "top_p":0.9,
        }

    chat = ChatBedrock(   # new chat model
        model_id=modelId,
        client=boto3_bedrock, 
        model_kwargs=parameters,
        region_name=region
    )
    
    if multi_region=='Enable':
        selected_chat = selected_chat + 1
        if selected_chat == number_of_models:
            selected_chat = 0
    else:
        selected_chat = 0

    return chat

def translate_text(text):
    chat = get_chat(extended_thinking=reasoning_mode)

    system = (
        "You are a helpful assistant that translates {input_language} to {output_language} in <article> tags. Put it in <result> tags."
    )
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)
    
    if isKorean(text)==False :
        input_language = "English"
        output_language = "Korean"
    else:
        input_language = "Korean"
        output_language = "English"
                        
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "input_language": input_language,
                "output_language": output_language,
                "text": text,
            }
        )
        msg = result.content
        logger.info(f"translated text: {msg}")
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")      
        raise Exception ("Not able to request to LLM")

    return msg[msg.find('<result>')+8:len(msg)-9] # remove <result> tag
    
def check_grammer(text):
    chat = get_chat(extended_thinking=reasoning_mode)

    if isKorean(text)==True:
        system = (
            "다음의 <article> tag안의 문장의 오류를 찾아서 설명하고, 오류가 수정된 문장을 답변 마지막에 추가하여 주세요."
        )
    else: 
        system = (
            "Here is pieces of article, contained in <article> tags. Find the error in the sentence and explain it, and add the corrected sentence at the end of your answer."
        )
        
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)
    
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "text": text
            }
        )
        
        msg = result.content
        logger.info(f"result of grammer correction: {msg}")
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")       
        raise Exception ("Not able to request to LLM")
    
    return msg

reference_docs = []

def isKorean(text):
    # check korean
    pattern_hangul = re.compile('[\u3131-\u3163\uac00-\ud7a3]+')
    word_kor = pattern_hangul.search(str(text))
    # print('word_kor: ', word_kor)

    if word_kor and word_kor != 'None':
        # logger.info(f"Korean: {word_kor}")
        return True
    else:
        # logger.info(f"Not Korean:: {word_kor}")
        return False
    
def traslation(chat, text, input_language, output_language):
    system = (
        "You are a helpful assistant that translates {input_language} to {output_language} in <article> tags." 
        "Put it in <result> tags."
    )
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)
    
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "input_language": input_language,
                "output_language": output_language,
                "text": text,
            }
        )
        
        msg = result.content
        # print('translated text: ', msg)
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")     
        raise Exception ("Not able to request to LLM")

    return msg[msg.find('<result>')+8:len(msg)-9] # remove <result> tag

def get_parallel_processing_chat(models, selected):
    global model_type
    profile = models[selected]
    region =  profile['region']
    modelId = profile['model_id']
    model_type = profile['model_type']
    maxOutputTokens = 4096
    logger.info(f'selected_chat: {selected}, region: {region}, modelId: {modelId}, model_type: {model_type}')

    if profile['model_type'] == 'nova':
        STOP_SEQUENCE = '"\n\n<thinking>", "\n<thinking>", " <thinking>"'
    elif profile['model_type'] == 'claude':
        STOP_SEQUENCE = "\n\nHuman:" 
    elif profile['model_type'] == 'openai':
        STOP_SEQUENCE = "" 
                          
    # bedrock   
    if aws_access_key and aws_secret_key:
        boto3_bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name=region,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            aws_session_token=aws_session_token,
            config=Config(
                retries = {
                    'max_attempts': 30
                }
            )
        )
    else:
        boto3_bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name=region,
            config=Config(
                retries = {
                    'max_attempts': 30
                }
            )
        )

    if profile['model_type'] != 'openai':
        parameters = {
            "max_tokens":maxOutputTokens,     
            "temperature":0.1,
            "top_k":250,
            "top_p":0.9,
            "stop_sequences": [STOP_SEQUENCE]
        }
    else:
        parameters = {
            "max_tokens":maxOutputTokens,     
            "temperature":0.1,
            "top_k":250,
            "top_p":0.9,
        }

    chat = ChatBedrock(   # new chat model
        model_id=modelId,
        client=boto3_bedrock, 
        model_kwargs=parameters,
    )        
    
    return chat


def show_extended_thinking(st, result):
    # logger.info(f"result: {result}")
    if "thinking" in result.response_metadata:
        if "text" in result.response_metadata["thinking"]:
            thinking = result.response_metadata["thinking"]["text"]
            st.info(thinking)

####################### LangChain #######################
# General Conversation
#########################################################
def general_conversation(query):
    # Initialize memory chain
    initiate()
    
    llm = get_chat(extended_thinking=reasoning_mode)

    system = (
        "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
        "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다." 
        "모르는 질문을 받으면 솔직히 모른다고 말합니다."
    )
    
    human = "Question: {input}"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system), 
        MessagesPlaceholder(variable_name="history"), 
        ("human", human)
    ])
                
    history = memory_chain.load_memory_variables({})["chat_history"]

    chain = prompt | llm | StrOutputParser()
    try: 
        stream = chain.stream(
            {
                "history": history,
                "input": query,
            }
        )  
        logger.info(f"stream: {stream}")
            
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")      
        raise Exception ("Not able to request to LLM: "+err_msg)
        
    return stream

def get_summary(docs):    
    llm = get_chat(extended_thinking=reasoning_mode)

    text = ""
    for doc in docs:
        text = text + doc
    
    if isKorean(text)==True:
        system = (
            "다음의 <article> tag안의 문장을 요약해서 500자 이내로 설명하세오."
        )
    else: 
        system = (
            "Here is pieces of article, contained in <article> tags. Write a concise summary within 500 characters."
        )
    
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)
    
    chain = prompt | llm    
    try: 
        result = chain.invoke(
            {
                "text": text
            }
        )
        
        summary = result.content
        logger.info(f"esult of summarization: {summary}")
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}") 
        raise Exception ("Not able to request to LLM")
    
    return summary

def summary_image(img_base64, instruction):      
    llm = get_chat(extended_thinking=reasoning_mode)

    if instruction:
        logger.info(f"instruction: {instruction}")
        query = f"{instruction}. <result> tag를 붙여주세요. 한국어로 답변하세요."
        
    else:
        query = "이미지가 의미하는 내용을 풀어서 자세히 알려주세요. markdown 포맷으로 답변을 작성합니다."
    
    messages = [
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}", 
                    },
                },
                {
                    "type": "text", "text": query
                },
            ]
        )
    ]
    
    for attempt in range(5):
        logger.info(f"attempt: {attempt}")
        try: 
            result = llm.invoke(messages)
            
            extracted_text = result.content
            # print('summary from an image: ', extracted_text)
            break
        except Exception:
            err_msg = traceback.format_exc()
            logger.info(f"error message: {err_msg}")                    
            raise Exception ("Not able to request to LLM")
        
    return extracted_text

def extract_text(img_base64):    
    multimodal = get_chat(extended_thinking=reasoning_mode)
    query = "텍스트를 추출해서 markdown 포맷으로 변환하세요. <result> tag를 붙여주세요."
    
    extracted_text = ""
    messages = [
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}", 
                    },
                },
                {
                    "type": "text", "text": query
                },
            ]
        )
    ]
    
    for attempt in range(5):
        logger.info(f"attempt: {attempt}")
        try: 
            result = multimodal.invoke(messages)
            
            extracted_text = result.content
            # print('result of text extraction from an image: ', extracted_text)
            break
        except Exception:
            err_msg = traceback.format_exc()
            logger.info(f"error message: {err_msg}")                    
            # raise Exception ("Not able to request to LLM")
    
    logger.info(f"Extracted_text: {extracted_text}")
    if len(extracted_text)<10:
        extracted_text = "텍스트를 추출하지 못하였습니다."    

    return extracted_text

fileId = uuid.uuid4().hex
# print('fileId: ', fileId)

def summarize_image(image_content, prompt, st):
    img = Image.open(BytesIO(image_content))
    
    width, height = img.size 
    logger.info(f"width: {width}, height: {height}, size: {width*height}")
    
    # 이미지 리사이징 및 크기 확인
    isResized = False
    max_size = 5 * 1024 * 1024  # 5MB in bytes
    
    # Initial resizing (based on pixel count)
    while(width*height > 2000000):  # Limit to approximately 2M pixels
        width = int(width/2)
        height = int(height/2)
        isResized = True
        logger.info(f"width: {width}, height: {height}, size: {width*height}")
    
    if isResized:
        img = img.resize((width, height))
    
    # Base64 size verification and additional resizing
    max_attempts = 5
    for attempt in range(max_attempts):
        buffer = BytesIO()
        img.save(buffer, format="PNG", optimize=True)
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        
        # Base64 size verification (actual transmission size)
        base64_size = len(img_base64.encode('utf-8'))
        logger.info(f"attempt {attempt + 1}: base64_size = {base64_size} bytes")
        
        if base64_size <= max_size:
            break
        else:
            # Resize smaller if still too large
            width = int(width * 0.8)
            height = int(height * 0.8)
            img = img.resize((width, height))
            logger.info(f"resizing to {width}x{height} due to size limit")
    
    if base64_size > max_size:
        logger.warning(f"Image still too large after {max_attempts} attempts: {base64_size} bytes")
        raise Exception(f"이미지 크기가 너무 큽니다. 5MB 이하의 이미지를 사용해주세요.")

    # extract text from the image
    if debug_mode=="Enable":
        status = "이미지에서 텍스트를 추출합니다."
        logger.info(f"status: {status}")
        st.info(status)

    text = extract_text(img_base64)
    logger.info(f"extracted text: {text}")

    if text.find('<result>') != -1:
        extracted_text = text[text.find('<result>')+8:text.find('</result>')] # remove <result> tag
        # print('extracted_text: ', extracted_text)
    else:
        extracted_text = text
    
    if debug_mode=="Enable":
        status = f"### 추출된 텍스트\n\n{extracted_text}"
        logger.info(f"status: {status}")
        st.info(status)
    
    if debug_mode=="Enable":
        status = "이미지의 내용을 분석합니다."
        logger.info(f"status: {status}")
        st.info(status)

    image_summary = summary_image(img_base64, prompt)
    
    if text.find('<result>') != -1:
        image_summary = image_summary[image_summary.find('<result>')+8:image_summary.find('</result>')]
    logger.info(f"image summary: {image_summary}")
            
    # if len(extracted_text) > 10:
    #     contents = f"## 이미지 분석\n\n{image_summary}\n\n## 추출된 텍스트\n\n{extracted_text}"
    # else:
    #     contents = f"## 이미지 분석\n\n{image_summary}"
    contents = f"## 이미지 분석\n\n{image_summary}"
    logger.info(f"image contents: {contents}")

    return contents

####################### Bedrock Agent #######################
# RAG using Lambda
############################################################# 
def get_rag_prompt(text):
    # print("###### get_rag_prompt ######")
    llm = get_chat(extended_thinking=reasoning_mode)
    # print('model_type: ', model_type)
    
    if model_type == "nova":
        if isKorean(text)==True:
            system = (
                "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
                "다음의 Reference texts을 이용하여 user의 질문에 답변합니다."
                "모르는 질문을 받으면 솔직히 모른다고 말합니다."
                "답변의 이유를 풀어서 명확하게 설명합니다."
            )
        else: 
            system = (
                "You will be acting as a thoughtful advisor."
                "Provide a concise answer to the question at the end using reference texts." 
                "If you don't know the answer, just say that you don't know, don't try to make up an answer."
                "You will only answer in text format, using markdown format is not allowed."
            )    
    
        human = (
            "Question: {question}"

            "Reference texts: "
            "{context}"
        ) 
        
    elif model_type == "claude":
        if isKorean(text)==True:
            system = (
                "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
                "다음의 <context> tag안의 참고자료를 이용하여 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다." 
                "모르는 질문을 받으면 솔직히 모른다고 말합니다."
                "답변의 이유를 풀어서 명확하게 설명합니다."
                "결과는 <result> tag를 붙여주세요."
            )
        else: 
            system = (
                "You will be acting as a thoughtful advisor."
                "Here is pieces of context, contained in <context> tags." 
                "If you don't know the answer, just say that you don't know, don't try to make up an answer."
                "You will only answer in text format, using markdown format is not allowed."
                "Put it in <result> tags."
            )    

        human = (
            "<question>"
            "{question}"
            "</question>"

            "<context>"
            "{context}"
            "</context>"
        )

    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)
    
    rag_chain = prompt | llm

    return rag_chain
 
bedrock_agent_runtime_client = boto3.client("bedrock-agent-runtime", region_name=region)
knowledge_base_id = config['knowledge_base_id']
number_of_results = 4
def retrieve(query):
    response = bedrock_agent_runtime_client.retrieve(
        retrievalQuery={"text": query},
        knowledgeBaseId=knowledge_base_id,
            retrievalConfiguration={
                "vectorSearchConfiguration": {"numberOfResults": number_of_results},
            },
        )
    
    # logger.info(f"response: {response}")
    retrieval_results = response.get("retrievalResults", [])
    # logger.info(f"retrieval_results: {retrieval_results}")

    json_docs = []
    for result in retrieval_results:
        text = url = name = None
        if "content" in result:
            content = result["content"]
            if "text" in content:
                text = content["text"]

        if "location" in result:
            location = result["location"]
            if "s3Location" in location:
                uri = location["s3Location"]["uri"] if location["s3Location"]["uri"] is not None else ""
                
                name = uri.split("/")[-1]
                # encoded_name = parse.quote(name)                
                # url = f"{path}/{doc_prefix}{encoded_name}"
                url = uri # TODO: add path and doc_prefix
                
            elif "webLocation" in location:
                url = location["webLocation"]["url"] if location["webLocation"]["url"] is not None else ""
                name = "WEB"

        json_docs.append({
            "contents": text,              
            "reference": {
                "url": url,                   
                "title": name,
                "from": "RAG"
            }
        })
    logger.info(f"json_docs: {json_docs}")

    return json.dumps(json_docs, ensure_ascii=False)
 
def run_rag_with_knowledge_base(query, st):
    global reference_docs, contentList
    reference_docs = []
    contentList = []

    # retrieve
    if debug_mode == "Enable":
        st.info(f"RAG 검색을 수행합니다. 검색어: {query}")  

    json_docs = retrieve(query)    
    logger.info(f"json_docs: {json_docs}")

    relevant_docs = json.loads(json_docs)

    relevant_context = ""
    for doc in relevant_docs:
        relevant_context += f"{doc['contents']}\n\n"

    # change format to document
    st.info(f"{len(relevant_docs)}개의 관련된 문서를 얻었습니다.")

    rag_chain = get_rag_prompt(query)
                       
    msg = ""    
    try: 
        result = rag_chain.invoke(
            {
                "question": query,
                "context": relevant_context                
            }
        )
        logger.info(f"result: {result}")

        msg = result.content        
        if msg.find('<result>')!=-1:
            msg = msg[msg.find('<result>')+8:msg.find('</result>')]        
               
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")                    
        raise Exception ("Not able to request to LLM")
    
    if relevant_docs:
        ref = "\n\n### Reference\n"
        for i, doc in enumerate(relevant_docs):
            page_content = doc["contents"][:100].replace("\n", "")
            ref += f"{i+1}. [{doc["reference"]['title']}]({doc["reference"]['url']}), {page_content}...\n"    
        logger.info(f"ref: {ref}")
        msg += ref
    
    return msg, reference_docs

def extract_thinking_tag(response, st):
    if response.find('<thinking>') != -1:
        status = response[response.find('<thinking>')+10:response.find('</thinking>')]
        logger.info(f"gent_thinking: {status}")
        
        if debug_mode=="Enable":
            st.info(status)

        if response.find('<thinking>') == 0:
            msg = response[response.find('</thinking>')+12:]
        else:
            msg = response[:response.find('<thinking>')]
        logger.info(f"msg: {msg}")
    else:
        msg = response

    return msg

streaming_index = None
index = 0
def add_notification(containers, message):
    global index

    if index == streaming_index:
        index += 1

    if containers is not None:
        containers['notification'][index].info(message)
    index += 1

def update_streaming_result(containers, message):
    global streaming_index
    streaming_index = index 

    if containers is not None:
        containers['notification'][streaming_index].markdown(message)

def update_tool_notification(containers, tool_index, message):
    if containers is not None:
        containers['notification'][tool_index].info(message)

tool_info_list = dict()
tool_input_list = dict()
tool_name_list = dict()

def get_tool_info(tool_name, tool_content):
    tool_references = []    
    urls = []
    content = ""

    # tavily
    if isinstance(tool_content, str) and "Title:" in tool_content and "URL:" in tool_content and "Content:" in tool_content:
        logger.info("Tavily parsing...")
        items = tool_content.split("\n\n")
        for i, item in enumerate(items):
            # logger.info(f"item[{i}]: {item}")
            if "Title:" in item and "URL:" in item and "Content:" in item:
                try:
                    title_part = item.split("Title:")[1].split("URL:")[0].strip()
                    url_part = item.split("URL:")[1].split("Content:")[0].strip()
                    content_part = item.split("Content:")[1].strip().replace("\n", "")
                    
                    logger.info(f"title_part: {title_part}")
                    logger.info(f"url_part: {url_part}")
                    logger.info(f"content_part: {content_part}")

                    content += f"{content_part}\n\n"
                    
                    tool_references.append({
                        "url": url_part,
                        "title": title_part,
                        "content": content_part[:100] + "..." if len(content_part) > 100 else content_part
                    })
                except Exception as e:
                    logger.info(f"Parsing error: {str(e)}")
                    continue                

    # OpenSearch
    elif tool_name == "SearchIndexTool": 
        if ":" in tool_content:
            extracted_json_data = tool_content.split(":", 1)[1].strip()
            try:
                json_data = json.loads(extracted_json_data)
                # logger.info(f"extracted_json_data: {extracted_json_data[:200]}")
            except json.JSONDecodeError:
                logger.info("JSON parsing error")
                json_data = {}
        else:
            json_data = {}
        
        if "hits" in json_data:
            hits = json_data["hits"]["hits"]
            if hits:
                logger.info(f"hits[0]: {hits[0]}")

            for hit in hits:
                text = hit["_source"]["text"]
                metadata = hit["_source"]["metadata"]
                
                content += f"{text}\n\n"

                filename = metadata["name"].split("/")[-1]
                # logger.info(f"filename: {filename}")
                
                content_part = text.replace("\n", "")
                tool_references.append({
                    "url": metadata["url"], 
                    "title": filename,
                    "content": content_part[:100] + "..." if len(content_part) > 100 else content_part
                })
                
        logger.info(f"content: {content}")
        
    # aws document
    elif tool_name == "search_documentation":
        try:
            json_data = json.loads(tool_content)
            for item in json_data:
                logger.info(f"item: {item}")
                
                if isinstance(item, str):
                    try:
                        item = json.loads(item)
                    except json.JSONDecodeError:
                        logger.info(f"Failed to parse item as JSON: {item}")
                        continue
                
                if isinstance(item, dict) and 'url' in item and 'title' in item:
                    url = item['url']
                    title = item['title']
                    content_text = item['context'][:100] + "..." if len(item['context']) > 100 else item['context']
                    tool_references.append({
                        "url": url,
                        "title": title,
                        "content": content_text
                    })
                else:
                    logger.info(f"Invalid item format: {item}")
                    
        except json.JSONDecodeError:
            logger.info(f"JSON parsing error: {tool_content}")
            pass

        logger.info(f"content: {content}")
        logger.info(f"tool_references: {tool_references}")
            
    # ArXiv
    elif tool_name == "search_papers" and "papers" in tool_content:
        try:
            json_data = json.loads(tool_content)

            papers = json_data['papers']
            for paper in papers:
                url = paper['url']
                title = paper['title']
                abstract = paper['abstract'].replace("\n", "")
                content_text = abstract[:100] + "..." if len(abstract) > 100 else abstract
                content += f"{content_text}\n\n"
                logger.info(f"url: {url}, title: {title}, content: {content_text}")

                tool_references.append({
                    "url": url,
                    "title": title,
                    "content": content_text
                })
        except json.JSONDecodeError:
            logger.info(f"JSON parsing error: {tool_content}")
            pass

        logger.info(f"content: {content}")
        logger.info(f"tool_references: {tool_references}")

    # aws-knowledge
    elif tool_name == "aws___read_documentation":
        logger.info(f"#### {tool_name} ####")
        if isinstance(tool_content, dict):
            json_data = tool_content
        elif isinstance(tool_content, list):
            json_data = tool_content
        else:
            json_data = json.loads(tool_content)
        
        logger.info(f"json_data: {json_data}")
        payload = json_data["response"]["payload"]
        if "content" in payload:
            payload_content = payload["content"]
            if "result" in payload_content:
                result = payload_content["result"]
                logger.info(f"result: {result}")
                if isinstance(result, str) and "AWS Documentation from" in result:
                    logger.info(f"Processing AWS Documentation format: {result}")
                    try:
                        # Extract URL from "AWS Documentation from https://..."
                        url_start = result.find("https://")
                        if url_start != -1:
                            # Find the colon after the URL (not inside the URL)
                            url_end = result.find(":", url_start)
                            if url_end != -1:
                                # Check if the colon is part of the URL or the separator
                                url_part = result[url_start:url_end]
                                # If the colon is immediately after the URL, use it as separator
                                if result[url_end:url_end+2] == ":\n":
                                    url = url_part
                                    content_start = url_end + 2  # Skip the colon and newline
                                else:
                                    # Try to find the actual URL end by looking for space or newline
                                    space_pos = result.find(" ", url_start)
                                    newline_pos = result.find("\n", url_start)
                                    if space_pos != -1 and newline_pos != -1:
                                        url_end = min(space_pos, newline_pos)
                                    elif space_pos != -1:
                                        url_end = space_pos
                                    elif newline_pos != -1:
                                        url_end = newline_pos
                                    else:
                                        url_end = len(result)
                                    
                                    url = result[url_start:url_end]
                                    content_start = url_end + 1
                                
                                # Remove trailing colon from URL if present
                                if url.endswith(":"):
                                    url = url[:-1]
                                
                                # Extract content after the URL
                                if content_start < len(result):
                                    content_text = result[content_start:].strip()
                                    # Truncate content for display
                                    display_content = content_text[:100] + "..." if len(content_text) > 100 else content_text
                                    display_content = display_content.replace("\n", "")
                                    
                                    tool_references.append({
                                        "url": url,
                                        "title": "AWS Documentation",
                                        "content": display_content
                                    })
                                    content += content_text + "\n\n"
                                    logger.info(f"Extracted URL: {url}")
                                    logger.info(f"Extracted content length: {len(content_text)}")
                    except Exception as e:
                        logger.error(f"Error parsing AWS Documentation format: {e}")
        logger.info(f"content: {content}")
        logger.info(f"tool_references: {tool_references}")

    else:        
        try:
            if isinstance(tool_content, dict):
                json_data = tool_content
            elif isinstance(tool_content, list):
                json_data = tool_content
            else:
                json_data = json.loads(tool_content)
            
            logger.info(f"json_data: {json_data}")
            if isinstance(json_data, dict) and "path" in json_data:  # path
                path = json_data["path"]
                if isinstance(path, list):
                    for url in path:
                        urls.append(url)
                else:
                    urls.append(path)            

            if isinstance(json_data, dict):
                for item in json_data:
                    logger.info(f"item: {item}")
                    if "reference" in item and "contents" in item:
                        url = item["reference"]["url"]
                        title = item["reference"]["title"]
                        content_text = item["contents"][:100] + "..." if len(item["contents"]) > 100 else item["contents"]
                        tool_references.append({
                            "url": url,
                            "title": title,
                            "content": content_text
                        })
            else:
                logger.info(f"json_data is not a dict: {json_data}")

                for item in json_data:
                    if "reference" in item and "contents" in item:
                        url = item["reference"]["url"]
                        title = item["reference"]["title"]
                        content_text = item["contents"][:100] + "..." if len(item["contents"]) > 100 else item["contents"]
                        tool_references.append({
                            "url": url,
                            "title": title,
                            "content": content_text
                        })
                
            logger.info(f"tool_references: {tool_references}")

        except json.JSONDecodeError:
            pass

    return content, urls, tool_references

async def run_strands_agent(query, strands_tools, mcp_servers, history_mode, containers):
    global tool_list, index
    tool_list = []
    index = 0

    image_url = []
    references = []
    
    # initiate agent
    await strands_agent.initiate_agent(
        system_prompt=None, 
        strands_tools=strands_tools, 
        mcp_servers=mcp_servers, 
        historyMode=history_mode
    )
    logger.info(f"tool_list: {tool_list}")    

    # run agent    
    final_result = current = ""
    with strands_agent.mcp_manager.get_active_clients(mcp_servers) as _:
        agent_stream = strands_agent.agent.stream_async(query)
        
        async for event in agent_stream:
            text = ""            
            if "data" in event:
                text = event["data"]
                logger.info(f"[data] {text}")
                current += text
                update_streaming_result(containers, current)

            elif "result" in event:
                final = event["result"]                
                message = final.message
                if message:
                    content = message.get("content", [])
                    result = content[0].get("text", "")
                    logger.info(f"[result] {result}")
                    final_result = result

            elif "current_tool_use" in event:
                current_tool_use = event["current_tool_use"]
                logger.info(f"current_tool_use: {current_tool_use}")
                name = current_tool_use.get("name", "")
                input = current_tool_use.get("input", "")
                toolUseId = current_tool_use.get("toolUseId", "")

                text = f"name: {name}, input: {input}"
                logger.info(f"[current_tool_use] {text}")

                if toolUseId not in tool_info_list: # new tool info
                    index += 1
                    current = ""
                    logger.info(f"new tool info: {toolUseId} -> {index}")
                    tool_info_list[toolUseId] = index
                    tool_name_list[toolUseId] = name
                    add_notification(containers, f"Tool: {name}, Input: {input}")
                else: # overwrite tool info if already exists
                    logger.info(f"overwrite tool info: {toolUseId} -> {tool_info_list[toolUseId]}")
                    containers['notification'][tool_info_list[toolUseId]].info(f"Tool: {name}, Input: {input}")

            elif "message" in event:
                message = event["message"]
                logger.info(f"[message] {message}")

                if "content" in message:
                    content = message["content"]
                    logger.info(f"tool content: {content}")
                    if "toolResult" in content[0]:
                        toolResult = content[0]["toolResult"]
                        toolUseId = toolResult["toolUseId"]
                        toolContent = toolResult["content"]
                        toolResult = toolContent[0].get("text", "")
                        tool_name = tool_name_list[toolUseId]
                        logger.info(f"[toolResult] {toolResult}, [toolUseId] {toolUseId}")
                        add_notification(containers, f"Tool Result: {str(toolResult)}")

                        content, urls, refs = get_tool_info(tool_name, toolResult)
                        if refs:
                            for r in refs:
                                references.append(r)
                            logger.info(f"refs: {refs}")
                        if urls:
                            for url in urls:
                                image_url.append(url)
                            logger.info(f"urls: {urls}")

                        if content:
                            logger.info(f"content: {content}")                
                
            elif "contentBlockDelta" or "contentBlockStop" or "messageStop" or "metadata" in event:
                pass

            else:
                logger.info(f"event: {event}")

        if references:
            ref = "\n\n### Reference\n"
            for i, reference in enumerate(references):
                page_content = reference['content'][:100].replace("\n", "")
                ref += f"{i+1}. [{reference['title']}]({reference['url']}), {page_content}...\n"    
            final_result += ref

        if containers is not None:
            containers['notification'][index].markdown(final_result)

    return final_result, image_url

async def run_langgraph_agent(query, mcp_servers, history_mode, containers):
    global index
    index = 0

    image_url = []
    references = []

    mcp_json = mcp_config.load_selected_config(mcp_servers)
    logger.info(f"mcp_json: {mcp_json}")

    server_params = langgraph_agent.load_multiple_mcp_server_parameters(mcp_json)
    logger.info(f"server_params: {server_params}")    

    try:
        client = MultiServerMCPClient(server_params)
        logger.info(f"MCP client created successfully")
        
        tools = await client.get_tools()
        logger.info(f"get_tools() returned: {tools}")
        
        if tools is None:
            logger.error("tools is None - MCP client failed to get tools")
            tools = []
        
        tool_list = [tool.name for tool in tools] if tools else []
        logger.info(f"tool_list: {tool_list}")
        
    except Exception as e:
        logger.error(f"Error creating MCP client or getting tools: {e}")
        
        # Try to refresh bearer token and retry
        logger.info("Attempting to refresh bearer token and retry...")
        try:
            # Get fresh bearer token from Cognito
            config = utils.load_config()
            bearer_token = mcp_config.create_cognito_bearer_token(config)
            if bearer_token:
                logger.info("Successfully obtained fresh bearer token")
                
                # Update the server parameters with new bearer token
                for server_name, server_config in server_params.items():
                    if 'headers' in server_config and 'Authorization' in server_config['headers']:
                        server_config['headers']['Authorization'] = f"Bearer {bearer_token}"
                        logger.info(f"Updated bearer token for server: {server_name}")
                
                # Save the new bearer token
                secret_name = config.get('secret_name')
                mcp_config.save_bearer_token(secret_name, bearer_token)
                
                # Retry with new bearer token
                logger.info("Retrying MCP client creation with fresh bearer token...")
                client = MultiServerMCPClient(server_params)
                tools = await client.get_tools()
                
                if tools is None:
                    logger.error("tools is still None after bearer token refresh")
                    tools = []
                
                tool_list = [tool.name for tool in tools] if tools else []
                logger.info(f"tool_list after retry: {tool_list}")
                
            else:
                logger.error("Failed to get fresh bearer token from Cognito")
                tools = []
                tool_list = []
                
        except Exception as retry_error:
            logger.error(f"Error during bearer token refresh and retry: {retry_error}")
            tools = []
            tool_list = []        

    # If no tools available, use general conversation
    if not tools:
        logger.warning("No tools available, using general conversation mode")
        result = "MCP 설정을 확인하세요."
        if containers is not None:
            containers['notification'][0].markdown(result)
        return result, image_url
    
    if history_mode == "Enable":
        app = langgraph_agent.buildChatAgentWithHistory(tools)
        config = {
            "recursion_limit": 50,
            "configurable": {"thread_id": user_id},
            "tools": tools,
            "system_prompt": None
        }
    else:
        app = langgraph_agent.buildChatAgent(tools)
        config = {
            "recursion_limit": 50,
            "configurable": {"thread_id": user_id},
            "tools": tools,
            "system_prompt": None
        }        
    
    inputs = {
        "messages": [HumanMessage(content=query)]
    }
            
    result = ""
    tool_used = False  # Track if tool was used
    tool_name = toolUseId = ""
    async for output in app.astream(inputs, config, stream_mode="messages"):
        # logger.info(f"output: {output}")

        # Handle tuple output (message, metadata)
        if isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], AIMessageChunk):
            message = output[0]    
            input = {}        
            if isinstance(message.content, list):
                for content_item in message.content:
                    if isinstance(content_item, dict):
                        if content_item.get('type') == 'text':
                            text_content = content_item.get('text', '')
                            # logger.info(f"text_content: {text_content}")
                            
                            # If tool was used, start fresh result
                            if tool_used:
                                result = text_content
                                tool_used = False
                            else:
                                result += text_content
                                
                            # logger.info(f"result: {result}")                
                            update_streaming_result(containers, result)

                        elif content_item.get('type') == 'tool_use':
                            logger.info(f"content_item: {content_item}")      
                            if 'id' in content_item and 'name' in content_item:
                                toolUseId = content_item.get('id', '')
                                tool_name = content_item.get('name', '')
                                logger.info(f"tool_name: {tool_name}, toolUseId: {toolUseId}")
                                add_notification(containers, f"Tool: {tool_name}, Input: {input}")

                                tool_info_list[toolUseId] = index                     
                                tool_name_list[toolUseId] = tool_name     
                                                                    
                            if 'partial_json' in content_item:
                                partial_json = content_item.get('partial_json', '')
                                logger.info(f"partial_json: {partial_json}")
                                
                                if toolUseId not in tool_input_list:
                                    tool_input_list[toolUseId] = ""                                
                                tool_input_list[toolUseId] += partial_json
                                input = tool_input_list[toolUseId]
                                logger.info(f"input: {input}")

                                logger.info(f"tool_name: {tool_name}, input: {input}, toolUseId: {toolUseId}")
                                # add_notification(containers, f"Tool: {tool_name}, Input: {input}")
                                index = tool_info_list[toolUseId]
                                containers['notification'][index-1].info(f"Tool: {tool_name}, Input: {input}")
                        
        elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], ToolMessage):
            message = output[0]
            logger.info(f"ToolMessage: {message.name}, {message.content}")
            tool_name = message.name
            toolResult = message.content
            toolUseId = message.tool_call_id
            logger.info(f"toolResult: {toolResult}, toolUseId: {toolUseId}")
            add_notification(containers, f"Tool Result: {toolResult}")
            tool_used = True
            
            content, urls, refs = get_tool_info(tool_name, toolResult)
            if refs:
                for r in refs:
                    references.append(r)
                logger.info(f"refs: {refs}")
            if urls:
                for url in urls:
                    image_url.append(url)
                logger.info(f"urls: {urls}")

            if content:
                logger.info(f"content: {content}")        
    
    if not result:
        result = "답변을 찾지 못하였습니다."        
    logger.info(f"result: {result}")

    if references:
        ref = "\n\n### Reference\n"
        for i, reference in enumerate(references):
            page_content = reference['content'][:100].replace("\n", "")
            ref += f"{i+1}. [{reference['title']}]({reference['url']}), {page_content}...\n"    
        result += ref
    
    if containers is not None:
        containers['notification'][index].markdown(result)
    
    return result, image_url
