import os
import torch

import chainlit as cl
from chainlit.input_widget import Select, Slider
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
#from sql_alchemy_custom import SQLAlchemyDataLayer
from llama_index.llms.ollama import Ollama as llamaindex_ollama
from llama_index.core.llms import ChatMessage
import ollama

from chainlit.types import ThreadDict
import time

from intelligence import *
from typing import Dict, Optional

from dotenv import load_dotenv
import os
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex

from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
)
 #https://docs.llamaindex.ai/en/stable/examples/vector_stores/postgres/
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import StorageContext


from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

load_dotenv()

hf_token = os.getenv("HF_TOKEN", None) 
model_id = os.getenv("model_id", None) 
embed_model_name = os.getenv("embed_model_name", None)
llama_model_id = os.getenv("llama_model_id", None)

host = os.getenv("host", None)
user = os.getenv("user", None)
password = os.getenv("password", None)
port = os.getenv("port", None)
database= os.getenv("database", None)
table_name= os.getenv("table_name", None)

BUCKET_NAME = os.getenv("BUCKET_NAME","STORAGE") 
AGENT_RUN = os.getenv("AGENT_RUN", None)
TOOLS_METHOD  =  os.getenv("TOOLS_METHOD", None)
conninfo = f"postgresql+asyncpg://{user}:{password}@{host}:5432/chainlit"

from FSStorageClient import *
from local_bucket import *

os.environ["TOKENIZERS_PARALLELISM"]="true"
############### Create a virtual storage client #########################

# Configure data layer
#https://github.com/Chainlit/chainlit/issues/1205
fs_storage_client = FSStorageClient(
    storage_path=os.path.join(os.getcwd(), BUCKET_NAME),
    url_path=BUCKET_NAME
)
cl.data._data_layer = SQLAlchemyDataLayer(conninfo=conninfo, storage_provider=fs_storage_client)

from llama_index.core.bridge.pydantic import BaseModel, Field
from typing import Optional
import chainlit as cl

from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.agent import ReActAgent

embedding_model = HuggingFaceEmbedding(model_name=embed_model_name)

#from ai_agent import *
from ai_custom_multiagent import *

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Fetch the user matching username from your database
    # and compare the hashed password with the value stored in the database
    if (username, password) == ("admin", "admin") : # Just for testing locally. Ideally this should not be hardcoded
        return cl.User(
            identifier=username, metadata={"role": username, "provider": "credentials"}
        )
    else:
        return None


@cl.on_chat_start
async def start():
    print("Starting conversation...")
    task_list = cl.TaskList()

    # Foundation model
    try:
        llm = HuggingFaceInferenceAPI(
            model_name=model_id, 
            tokenizer_name=model_id, 
            token=hf_token,
            context_window = 100000,
            num_outputs = Field(
                    default=4000,
                    description="The number of tokens to generate.",
                ),
            is_chat_model = True,
            generate_kwargs={
                "do_sample": True,
                "temperature": 0,
                "top_p": 0.9,
            },
        )

        cl.user_session.set("model", ollama)


        # Embedding model
        cl.user_session.set("embed_model", embedding_model)
        
        # Configs
        
        # # bge embedding model
        Settings.embed_model = embedding_model

        # # Llama-3-8B-Instruct model
        Settings.llm = llm

        pgvector_store = PGVectorStore.from_params(
            database = database,
            host = host,
            password = password,
            port = port,
            user = user,
            table_name = table_name, 
            embed_dim = 1024,  # selected model embedding dimension
            hybrid_search=True, # Hybrid vector search

        )

        index = VectorStoreIndex.from_vector_store(
            vector_store=pgvector_store, 
            use_async = True
            )

        semantic_retriever = index.as_retriever(similarity_top_k=40)
        
        # sparse_retriever= index.as_retriever(
        #     vector_store_query_mode="sparse", sparse_top_k=10
        #     )

        # fusion_retriever = QueryFusionRetriever(
        #         [semantic_retriever],#, sparse_retriever],
        #         similarity_top_k=20,
        #         num_queries=2,  # set this to 1 to disable query generation
        #         mode="relative_score",
        #         use_async=True,
        #     )
        # response_synthesizer = CompactAndRefine()
        # query_engine = RetrieverQueryEngine(
        #     retriever=fusion_retriever,
        #     response_synthesizer=response_synthesizer,
        # )
        retriever = semantic_retriever
        # Use the index as a retriever
        
        cl.user_session.set("retriever", retriever)

        cl.user_session.set("answer_sources", [])
        cl.user_session.set("conversation_history", [])
        cl.user_session.set("resume_history", [])
        cl.user_session.set("user_input", "")
        cl.user_session.set("temperature_param", 0)
        cl.user_session.set("language", "PT-PT")
        cl.user_session.set("most_recent_history",[])
        cl.user_session.set("message_history", [])
        cl.user_session.set("task_list", task_list)

        if AGENT_RUN :
            
            agent = AIAgent(
                retriever=retriever,
                agent_enabled = AGENT_RUN,
                tools_method = "ollama",
            )
            cl.user_session.set("agent", agent)
        
    except Exception as e:
        print("Erro init variables: ", e)

    # Settings
    settings = await cl.ChatSettings(
        [
            Select(
                id="Language",
                label="Language",
                values=["PT", "EN"],
                initial_index=0,
            ),
            Slider(
                id="Temperature",
                label="Temperature",
                initial=0,
                min=0,
                max=1,
                step=0.05,
            ),
        ]
    ).send()

    msg = cl.Message(
        author="Assistant", content="Olá, sou o teu Assistente Virtual pessoal. Em que posso ajudar ?"
    )
    await msg.send()

    update_conversation_history(msg)
    sources_update(msg.elements)


@cl.on_settings_update
async def setup_agent(settings):

    print("on_settings_update: ", settings)

    if settings["Language"] == "PT":
        cl.user_session.set("language", "PT-PT")
    elif settings["Language"] == "EN":
        cl.user_session.set("language", "EN-GB")

    if float(settings["Temperature"]) !=  float(cl.user_session.get("temperature_param")):
        print(f"Temperature parameter changed from {float(cl.user_session.get('temperature_param'))} to {float(settings['Temperature'])}")
        # Update temperature
        cl.user_session.set("temperature_param", settings["Temperature"])
               
        llm = llamaindex_ollama(
            model="llama3.1:latest", 
            request_timeout=120.0, 
            temperature = cl.user_session.get("temperature_param"), 
            context_window = 100000)
        cl.user_session.set("model", llm)

        if AGENT_RUN :
            agent = cl.user_session.get("agent")
            agent.llm = llm
            cl.user_session.set("agent", agent)


@cl.action_callback("Sources")
async def on_action(action):

    print("Action: ", action)
    count = 0
    while count < 10:
        chat = cl.user_session.get("conversation_history")
        try:
            print("chat length | action +1  | Count",len(chat), int(action.value) +1, count)
            if len(chat)> int(action.value):
                break
        except Exception as e:
            print(f"Failed to get chat history on action. {e}")
            
        time.sleep(1)
        count +=1
    
    print("on_action()  chat history and length: ",chat, len(chat))
    
    msg = chat[int(action.value)]
    
    print("on_action()  chat message content : ", msg.content)
    
    if len( msg.elements ) == 0:
        print("answer_sources and length : ",cl.user_session.get("answer_sources"), len(cl.user_session.get("answer_sources")))

        msg.elements = cl.user_session.get("answer_sources")[int(action.value)]
    else:
        msg.elements = []
    chat[int(action.value)] = msg
    
    cl.user_session.set("conversation_history", chat)
    
    await msg.update()
    
    await action.remove()

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):

    llm = HuggingFaceInferenceAPI(
        model_name=model_id, 
        tokenizer_name=model_id, 
        token=hf_token,
        context_window = 100000,
        num_outputs = Field(
                default=4000,
                description="The number of tokens to generate.",
            ),
        is_chat_model = True,
        generate_kwargs={
            "do_sample": True,
            "temperature": 0,
            "top_p": 0.9,
        },
    )

    cl.user_session.set("model", ollama)


    # Embedding model
    embedding_model = HuggingFaceEmbedding(model_name=embed_model_name)
    cl.user_session.set("embed_model", embedding_model)
    
    # Configs
    
    # # bge embedding model
    Settings.embed_model = embedding_model

    # # Llama-3-8B-Instruct model
    Settings.llm = llm

    pgvector_store = PGVectorStore.from_params(
        database = database,
        host = host,
        password = password,
        port = port,
        user = user,
        table_name = table_name, 
        embed_dim = 1024,  # selected model embedding dimension
        hybrid_search=True, # Hybrid vector search

    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=pgvector_store, 
        use_async = True
        )

    semantic_retriever = index.as_retriever(similarity_top_k=30)
    
    sparse_retriever= index.as_retriever(
        vector_store_query_mode="sparse", sparse_top_k=30
        )

    fusion_retriever = QueryFusionRetriever(
            [semantic_retriever, sparse_retriever],
            similarity_top_k=30,
            num_queries=10,  # set this to 1 to disable query generation
            mode="relative_score",
            use_async=True,
        )
    response_synthesizer = CompactAndRefine()
    query_engine = RetrieverQueryEngine(
        retriever=fusion_retriever,
        response_synthesizer=response_synthesizer,
    )
    retriever = query_engine
    # Use the index as a retriever

    if AGENT_RUN :
        
        agent = AIAgent(
            retriever=retriever,
            agent_enabled = AGENT_RUN,
            tools_method = "ollama",
        )
        cl.user_session.set("agent", agent)
        

    # Restore conversation info
    cl.user_session.set("answer_sources", [])
    cl.user_session.set("conversation_history", [])
    cl.user_session.set("resume_history", [])
    cl.user_session.set("user_input", "")
    cl.user_session.set("language", thread["metadata"]["language"])
    cl.user_session.set("temperature_param", thread["metadata"]["temperature_param"])
    cl.user_session.set("most_recent_history", thread["metadata"]["most_recent_history"])
    cl.user_session.set("message_history", thread["metadata"]["message_history"])


    llm = llamaindex_ollama(
            model=f"{llama_model_id}:latest", 
            request_timeout=120.0, 
            temperature = thread["metadata"]["temperature_param"], 
            context_window = 100000)

    cl.user_session.set("model", llm)
    Settings.llm = llm
    Settings.embed_model = embedding_model
    
    # pgvector_store = PGVectorStore.from_params(
    #         database = database,
    #         host = host,
    #         password = password,
    #         port = port,
    #         user = user,
    #         table_name = table_name, 
    #         embed_dim = 1024,  # selected model embedding dimension
    #         text_search_config = "english",
    #         hnsw_kwargs={
    #             "hnsw_m": 16,
    #             "hnsw_ef_construction": 64,
    #             "hnsw_ef_search": 40,
    #             "hnsw_dist_method": "vector_cosine_ops",
    #         },
    # )

    # index = VectorStoreIndex.from_vector_store(
    #     vector_store=pgvector_store, 
    #     vector_store_kwargs={
    #         "mmr_threshold": 0.7 
    #         }
    #     )

    # # chat_engine = index.as_chat_engine()
    # # cl.user_session.set("chat_engine", chat_engine)
    
    # # Use the index as a retriever
    # retriever = index.as_retriever(similarity_top_k=50)
    
    if AGENT_RUN :
        agent = AIAgent(
            llm = llm,
            agent_enabled = AGENT_RUN,
            tools_method = "ollama",
        )
        cl.user_session.set("agent", agent)
    
    for i in range(len(thread["metadata"]["resume_history"])):
        
        update_conversation_history(
            cl.Message(content = thread["metadata"]["resume_history"][i], author="Assistant")
            )

        sources_update([])
    
    settings = await cl.ChatSettings(
        [
            Select(
                id="Language",
                label="Language",
                values=["PT", "EN"],
                initial_index=0,
            ),
            Slider(
                id="Temperature",
                label="Temperature",
                initial=thread["metadata"]["temperature_param"],
                min=0,
                max=1,
                step=0.05,
            ),
        ]
    ).send()

@cl.on_message
async def main(message: cl.Message):
    
    conversation_history = cl.user_session.get("conversation_history")
    history = []
    n_historic_data = 4
    for i,hist_msg in enumerate(conversation_history):
        if i%2:
            history.append("User: " + hist_msg.content)
        else:
            history.append("Assistant: " + hist_msg.content)
    
    history = history[-n_historic_data:]
    cl.user_session.set("most_recent_history", history)
    print(f"Last N interactions : {history}")
    
    update_conversation_history(message)
    sources_update(message.elements) #for now empty - just to assure the array has the same lenght as chat history

    print("New Message: ", message.content)

    cl.user_session.set("user_input", message.content)
    # TODO : Get intent, get speficif year of filter, etc... TOOLS 

    print("Getting response")

    if AGENT_RUN :
        agent = cl.user_session.get("agent")
        res = await agent.retrieve_and_generate(history, message.content)
        context = cl.user_session.get("answer_sources")[int(len(cl.user_session.get("conversation_history")))]

    else:
        context = []
        model = cl.user_session.get("model")

        if search_intent(model, message.content):
            print("Intent : Search")
            context = retrieve_chunks(message.content)
        else :
            sources_update([])
            print("Intent : Conversational")
        res = retrieve_and_generate(history, message.content, context)
    
    
    # Process sources along with the message
    if len(context)> 0 :
        msg  = cl.Message(content="",  author="Assistant", actions = [cl.Action(name="Sources", value= str(len(cl.user_session.get("conversation_history"))))])
    else:
        msg = cl.Message(content="",  author="Assistant") 
    
    # Stream the output message
    for token in res:
        try:
            await msg.stream_token(token)
        except Exception as e:
            print(f"Failed to retrieve the token {token}. {e}")
    
    await msg.send()
    
    # Update the conversation history
    update_conversation_history(msg)
    

@cl.on_chat_end
def end():
    print("Finalizing the conversation...")

    # cl.user_session.set("model", None)
    # cl.user_session.set("embed_model", None)

    # Settings.embed_model = None
    # Settings.llm = None

    # cl.user_session.set("agent", None)

    # cl.user_session.set("retriever", None)

    # cl.user_session.set("answer_sources", [])
    # cl.user_session.set("conversation_history", [])
    # cl.user_session.set("resume_history", [])
    # cl.user_session.set("user_input", "")
    # cl.user_session.set("temperature_param", 0)
    # cl.user_session.set("language", "PT-PT")
    # cl.user_session.set("most_recent_history",[])