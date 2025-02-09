import chainlit as cl
import copy
import re
import os
from llama_index.core.postprocessor import LLMRerank
from llama_index.core import QueryBundle
from llama_index.core.tools import FunctionTool
from llama_index.core.tools import QueryEngineTool
from typing import List, Dict, Any
from functools import wraps
from llama_index.llms.ollama import Ollama as llamaindex_ollama
from llama_index.core.llms import ChatMessage
import ollama

from urllib.parse import quote
from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
)
from llama_index.core import Settings, VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.agent import FunctionCallingAgentWorker
from langchain_ollama import ChatOllama

from llama_agents import (
    AgentService,
    ControlPlaneServer,
    SimpleMessageQueue,
    AgentOrchestrator,
)
from llama_agents import LocalLauncher
from llama_index.core.agent import FunctionCallingAgentWorker
import time
import psycopg2
from intelligence import sources_update
from prompts import *


host = os.getenv("host", None)
user = os.getenv("user", None)
password = os.getenv("password", None)
port = os.getenv("port", None)
database= os.getenv("database", None)
table_name= os.getenv("table_name", None)
llama_model_id = os.getenv("llama_model_id", None)
supervisor_llm = os.getenv("supervisor_model_id", None)
worker_llm = os.getenv("work_model_id", None)
from typing import Literal
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

from langgraph.graph import MessagesState, END
from langgraph.types import Command

from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic


from ollama import chat
from pydantic import BaseModel
import json 

DOCUMENT_ROOT_PATH = "./DOCUMENTS/"
os.makedirs(DOCUMENT_ROOT_PATH, exist_ok=True)
print(f"Directory '{DOCUMENT_ROOT_PATH}' created or already exists.")


members = [
    "intro_creator",
    "body_creator",
    "reference_creator"
    ]

class MultiAgentSupervisor():
    def __init__(
            self,
            model,
            **kwargs
            ):
        
        self.agent = model
        self.messages_list = []
   
    def invoke(self, state, options , system_prompt ):
        class NextWorker(BaseModel):
            next: Literal[*options]
        
        prompt = system_prompt

        self.messages_list.append(
            {
            'role': 'user',
            'content': prompt
            }
        )
        #https://github.com/ollama/ollama/blob/main/docs/api.md?plain=1
        response = self.agent.chat(
            messages=self.messages_list,
            model=llama_model_id,
            format=NextWorker.model_json_schema(),
        )

        next_worker = NextWorker.model_validate_json(response.message.content)

        self.messages_list.append(response.message)
        print(f"Supervisor output : '{next_worker.next}'")
        print("\n----------------------------------\n")
        return next_worker.next


class MultiAgentWorkers():
    def __init__(
            self,
            worker_name, 
            model,
            prompt,
            language,
            **kwargs
            ):
        
        self.worker_name = worker_name
        self.agent = model
        self.prompt = prompt
        self.language = language
        self.message_list = []
        #print("\n-------------------------------------\n")
        #print(f"Worker {worker_name} : \nLanguage : {language} \nPrompt : {prompt}")
    
    
    def invoke(self, topic : str, state : {}, revise = False ) -> dict:
        
        if self.worker_name in state:
            _ = state.pop(self.worker_name)
        print("Other workers involved : ", state.keys())

        content = self.prompt.replace("$LANGUAGE$", self.language)+ f"\nHere you have the topic to write about: {topic}.\nHere you have the content from the other sections of the document: {state}"

        if revise:
            print(f"Revising {self.worker_name}")
            content = content+ "\n Revise and improve your previous output."
        #print(f"Worker {self.worker_name} prompt : ",content)
        
        self.message_list.append({
                    'role': 'user',
                    'content': content,
                })

        response = self.agent.chat(
            messages=self.message_list,
            model=llama_model_id,
            options = {
                "temperature": 0.9,
                "num_predict" : 8000
            }
        )
        self.message_list.append(response.message)

        print(f"{self.worker_name} response : ",response.message.content)
        return response.message.content

intent_options = ["SEARCH", "NORMAL", "DOC_WRITER"]

class UsertIntent(BaseModel):
    output: Literal[*intent_options]

class AIAgent():
    def __init__(
            self,
            #llm,
            agent_enabled,
            tools_method = "ollama",
            retriever = None,
            
            **kwargs
            ):
        self.llama_model_id = llama_model_id
        self.reranker =  LLMRerank(choice_batch_size=20, top_n=20)
        self.tools_method = tools_method
        self.agent_enabled = agent_enabled
        self.extra_params = kwargs
        self.agent = ollama
        self.intent_agent = ollama
        self.new_members = copy.deepcopy(members)
        self.supervisor = MultiAgentSupervisor(model = ollama )

        self.workers = {name : MultiAgentWorkers(worker_name = name, model = ollama, prompt = workers_instructions[name], language = cl.user_session.get("language")) for name in self.new_members}
        self.retriever = retriever
        self.worker_finished = []
        self.running_worker = 0
  
    #https://stackoverflow.com/questions/78404535/langchain-python-with-structured-output-ollama-functions

    def supervisor_node(self, topic :str) -> dict:

        output_per_agent = {}

        workers = copy.deepcopy(self.new_members)
        #while True:
        for _ in range(6): #Retry 10 rounds of agents to provide a proper document
            options = workers + ["FINISH"]
   
            system_prompt =  [
                "Role: You are a supervisor tasked with managing a conversation between specialized workers to create a cohesive document.",
                "Your role is to oversee the workflow, evaluate worker outputs, and ensure each worker adheres to its assigned responsibilities.",
                "Workers and Their Responsibilities:",
            ]

            for m in workers:
                system_prompt.append(workers_prompt_intruction[m])

            system_prompt =  system_prompt +  [
                """Instructions:

                1. You will receive:
                - A user request asking for the creation of a document on a specific topic.
                - The outputs of the workers (if any).

                2. Your task is to:
                - Evaluate the workers' outputs. If a worker writes about a section that is not its responsibility, select the same worker again for refinement and respond with the worker to act next. If a worker writes about its assigned section, respond with the next worker to act.
                - Ensure the document is created in the correct order and that each section aligns with the overall topic and other sections.
                - When all sections are complete and meet the requirements, respond with FINISH.

                3. Workflow:
                - Each worker will perform its task and respond with its results.
                - You will evaluate the results and decide the next steps.

                4. Constraints:
                - Do not write any sections yourself. Your role is strictly to supervise and manage the workers.
                - Ensure the workers adhere to their assigned responsibilities and do not overlap into other sections.""",
                f"Here you have the topic to write about: {topic}.",
                f"Here you have the content from the other sections of the document: {output_per_agent}",
                
            ]


            system_prompt_str =  '\n'.join(system_prompt)
            print("options : ", options)
            next_agent = self.supervisor.invoke(output_per_agent, options, system_prompt_str)

            if next_agent in workers :
                revise = next_agent in output_per_agent.keys()
                worker = self.workers[next_agent]
                output_per_agent[next_agent] = worker.invoke(topic, output_per_agent, revise)
                #workers.remove(next_agent)

            elif next_agent == "FINISH":
                break

        print(f"Final answer from the workers {output_per_agent.keys()} is : {output_per_agent}' ")

        self.workers = {name : MultiAgentWorkers(worker_name = name, model = ollama, prompt = workers_instructions[name], language = cl.user_session.get("language")) for name in self.new_members}
        self.supervisor = MultiAgentSupervisor(model = ollama )

        return output_per_agent, [w for w in self.new_members if w in output_per_agent.keys() ]

    async def retrieve_chunks(self, user_input):

        query_result_list = await self.retriever.aretrieve(user_input)

        if len(query_result_list)> 20 and self.reranker is not None:

            # choice_batch_size parameter is used to specify the number of items processed at a time during reranking. 
            # When you have a large number of items to rerank, batching can improve efficiency by reducing memory usage or managing API call limits.

            # How it works:
            # Purpose: Divides the input data into smaller chunks (batches) for reranking. Each batch is processed independently before the results are combined.
            # Default Value: If not specified, the default value varies based on the implementation, but it's typically optimized for performance.
            # Use Case: You can set this to a smaller number if you're facing memory constraints or to a higher value to improve throughput.

            # configure reranker
            query_result_list = self.reranker.postprocess_nodes(
                query_result_list, 
                QueryBundle(user_input)
            )

        sources =  []
        ids_retrieved = []
        sources_elements = []

        print(" query_result_list -> ", len(query_result_list))
        if len(query_result_list)>0:
            for result in query_result_list:
                chunk_id = result.node.id_
                metadata= result.node.metadata
                score = result.score
                chunk_text = result.node.text

                if  chunk_id not in ids_retrieved:
                    ids_retrieved.append(chunk_id)
                    sources.append(chunk_text)
                    
                    # Sources to be presented
                    doc_name = metadata["file_path"].split("/")[-1].replace(".pdf","")
                    sources_elements.append(cl.Pdf(name=doc_name, path = metadata["file_path"], display="inline") )

        sources_update(sources_elements)
        print("Finished retriving chunks...")
        return sources

    def get_intent(self, user_input: str) -> str:
        """Getting the intent of the user's input to understand if the user intention is to search 
        for specific information from the knowledge base or just having a conversational/normal interaction.
        This should be the first step of the agent."""
        

        messages_list = [(
            {
            'role': 'user',
            'content': copy.deepcopy(INTENT_PROMPT).replace("$USER_INPUT$", user_input, ).replace("$HISTORY$", '\n'.join(cl.user_session.get("most_recent_history"))),
            }
        )]
        response = self.intent_agent.chat(
            messages=messages_list,
            model=llama_model_id,
            format=UsertIntent.model_json_schema(),
        )

        user_intent = UsertIntent.model_validate_json(response.message.content)

 
        if user_intent.output !="SEARCH" and user_intent.output.find('SEARCH') == -1 and user_intent.output !="DOC_WRITER" and user_intent.output.find('DOC_WRITER') == -1:  
            print("Filling empty sources as this is a question for not using RAG...")
            sources_update([])

        print("Intent detected : ", user_intent.output)
        return user_intent.output

    def get_intent_olama_tool(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_intent_tool",
                "description": """ Understand the user's intent. All subsequent steps rely on the intent identified by this tool.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_input": {
                            "type": "string",
                            "description": "The user input to be analyzed and classify the intent."
                        }
                    },
                    "required": ["user_input"]
                }
            }
        }

    @cl.step(type="tool", name = "Getting the input's intent ...")
    def get_intent_tool(self, user_input):
        return self.get_intent(user_input)

    async def retrieve_data(self, user_input: str) -> str:
        """In case the user's intent is 'SEARCH' the user's input is used for querying the knowledge base, in order to find chunks
        capable of answering the user's question. These chunks of information are returned as a search result to the agent to use them
        to answer the question."""
        
        sources = await self.retrieve_chunks(user_input)
        if len(sources)> 0 :
            return "The Search result is : "+ "\n".join(sources)
        else:
            return "No relevant data found."

    def retrieve_data_olama_tool(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "retrieve_data_tool",
                "description": """Query the knowledge base to find information to answer the question.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_input": {
                            "type": "string",
                            "description": "The user input to search in the knowledge base"
                        }
                    },
                    "required": ["user_input"]
                }
            }
        }

    @cl.step(type="tool", name = "Retrieving data from the knowledge base ...")
    async def retrieve_data_tool(self, user_input):
        """
        Asynchronously queries the knowledge base using the given user input to retrieve information
        capable of answering the user's question. This method wraps around the `retrieve_data` function
        to provide an async interface for data retrieval.
        
        Args:
            user_input (str): The user input to search in the knowledge base.
            
        Returns:
            str: The search result containing chunks of information gathered from the knowledge base,
            formatted as a string.
        """

        return await self.retrieve_data(user_input)

    @cl.step(type="tool", name = "Refining query ...")
    def get_query_improvement(self, user_input : str, extra_instructions : str = " ") -> str:
        query_refinement_prompt = copy.deepcopy(QUERY_OPTIMIZATION).replace("$USER_INPUT$", user_input).replace("$HISTORY$", '\n'.join(cl.user_session.get("most_recent_history"))).replace("$EXTRA_INSTRUCTIONS$", extra_instructions)
        
        resp = ollama.generate(model=self.llama_model_id, prompt=query_refinement_prompt)

        return resp['response']

    def translate_output(self, answer: str) -> str:
        """Translate the output to the specified language so that the user can understand it.
        The is the last step of the agent."""

        resp = ollama.generate(model=self.llama_model_id, prompt=copy.deepcopy(TRANSLATE_OUTPUT_PROMPT).replace("$LANGUAGE$",cl.user_session.get('language')))
        # if resp['response'] != "SEARCH":
        #     sources_update([])

        return resp['response']

    def translate_output_olama_tool(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "translate_output_tool",
                "description": f"""Translate the output to the language '{cl.user_session.get("language")},  ONLY if it is in a different language'.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "The answer to be translated to the specified language"
                        }
                    },
                    "required": ["answer"]
                }
            }
        }

    def evaluate_responses_relevancy(self, user_input: str, answer: str, search_result: str) -> str:

        eval_result_int = ollama.generate(model=self.llama_model_id, prompt=copy.deepcopy(RESPONSE_EVALUATION_PROMPT).replace("$USER_INPUT$", user_input).replace("$SEARCH_RESULT$", search_result).replace("$ANSWER$", answer))
        resp = eval_result_int['response'].strip().replace('\n','').replace("'","")
        
        print(f"Relevancy {resp} ")
        return int(resp)

    def create_doc_agents_tool(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "create_doc_tool",
                "description": f"""Create/Generate/Write a document about a specified topic/subject.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "The subject/topic of the document"
                        }
                    },
                    "required": ["topic"]
                }
            }
        }
    
    def evaluate_responses_tool(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "evaluate_responses_tool",
                "description": f"""Evaluate the answer generated by the agent before output it to the user.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "The answer generated by the agent to be evaluated"
                        }
                    },
                    "required": ["answer"]
                }
            }
        }
    
    @cl.step(type="tool", name = "Translating the output ...")
    def translate_output_tool(self, answer):
        return self.translate_output(answer)

    async def retrieve_and_generate(self, history, user_input ):

        if self.agent_enabled:
    
            final_query = copy.deepcopy(AGENT_SYSTEM_PROMPT)
            final_query = final_query.replace("$HISTORY$", str(history))
            final_query = final_query.replace("$LANGUAGE$", cl.user_session.get('language'))
            final_query = final_query.replace("$USER_INPUT$", user_input)

            
            # https://github.com/ollama/ollama-python/blob/main/examples/tools.py
            # https://ollama.com/blog/functions-as-tools
            
            message_history = cl.user_session.get("message_history")

            msg = {"role":"user", "content" : final_query}
            message_history.append(msg)
            messages = copy.deepcopy(message_history)
            

            intent_classifier_tool, retrieve_acquisition_tool, language_tool, evaluate_responses_tool, create_doc_agents_tool  = self.get_tools()

            response = self.agent.chat(
                    model=self.llama_model_id,
                    messages= messages,
                    tools = [
                            intent_classifier_tool, 
            
                        ]
                )
            
            called_tools = []
            intent = None
            tools_list = None
            refined_query = None
            search_res = None
            print("First model response ->",response.message)
            refine_again = False
            retry = 0
            queries_done = [user_input]
            output = ""
            info_provided = False
            while response.message.tool_calls:
                print("Tools : ", response.message.tool_calls)
            
                for tool in response.message.tool_calls:
                    print(f"Running tool: {tool.function.name} with parameters {tool.function.arguments}")
                    
                    called_tools.append(tool.function.name)

                    if tool.function.name == "get_intent_tool":
                        print( "Getting intent...")
                        intent = self.get_intent_tool(user_input)

                        if intent =="SEARCH" or intent.find('SEARCH') != -1:
                            output = f"The intent is '{intent}'"
                            tools_list = [retrieve_acquisition_tool ]

                        elif intent == "DOC_WRITER" or intent.find('DOC_WRITER') != -1:
                            output = f"The intent is '{intent}'. Create the document for the given topic using the tool 'create_doc_tool'." 
                            tools_list = [create_doc_agents_tool ]

                        elif intent == "NORMAL" or intent.find('NORMAL') != -1:
                            output = f"The intent is '{intent}'. There's no need to search in the knowledge base. Just engage in a conversation with the user and answer politely to the user's input."
                            tools_list = [language_tool ]
                        

                    elif tool.function.name =="create_doc_tool":

                        try:
                            topic_to_process = tool.function.arguments["topic"]
                        except : 
                            topic_to_process = user_input

                        print("Creating document with topic : ", topic_to_process)
                        answer, workers_processed = self.supervisor_node(topic_to_process)
                        print("workers_processed ->",workers_processed)
                        doc_text = ""
                        for m in workers_processed:
                            doc_text = doc_text + "\n\n" + answer[m]
              
                        doc_path = f"{DOCUMENT_ROOT_PATH}/"+topic_to_process+".txt"
                        with open(doc_path, "w") as file:
                            file.write(doc_text)

                        sources_update([cl.File(name = topic_to_process, path = doc_path, display="inline")])

                        output =  f"Here you have the document created : {doc_text}. Sumarize it as an output for the user and inform the user that the full document was created in the directory '{doc_path}', and it should access to the .txt file to acess the full text. "
                        tools_list = [language_tool]
                        info_provided = True

                    elif tool.function.name == "retrieve_data_tool":
                        print("Retriving data from the knowledge base ...")

                        if refine_again and refined_query is not None:
                            print("Refining the question again...")
                            extra_instructions = f"""Here you have the last few refinements done to the user's input, that didn't provided a good search result enough to answer the question : {queries_done}.Take this in consideration when refining the query this time."""
                            refined_query = self.get_query_improvement(user_input, extra_instructions)
                        else:
                            print("New question to be refined...")
                            refined_query = self.get_query_improvement(user_input)

                        queries_done.append(refined_query)
                        print("Refined query  : ", refined_query)
                        search_res = await self.retrieve_data_tool(refined_query)
                        output = str(search_res)+ "\n\nUsing this search result answer the question : " + user_input
                        tools_list = [language_tool,  create_doc_agents_tool]
                        info_provided = True

                    elif tool.function.name == "translate_output_tool":
                        print("Translating output...")
                        output = self.translate_output_tool(tool.function.arguments)
                        tools_list = None
                    
                    print("Calling the agent again")
                    messages.append(response.message)
                    messages.append({'role': 'tool', 'content': str(output), 'name': tool.function.name})

                    # Get final response from model with function outputs
                    print(f"Message created with output {output}. \nCalling the agent with tool : {tools_list}")
                    if tools_list is None:
                        print("Calling without tools")
                        response = self.agent.chat(
                            self.llama_model_id, 
                            messages=messages,

                            )
                    else:
                        print("Calling with tools")
                        response = self.agent.chat(
                            self.llama_model_id, 
                            messages=messages,
                            tools = tools_list
                            )
  
                    
                    print('Response:', response.message.content)
                    print("Response tools : ",response.message.tool_calls)
                    print("Retry : ",retry)
                    
                    if not response.message.tool_calls and retry < 2 and search_res is not None :
                        eval_result = self.evaluate_responses_relevancy(user_input, response.message.content, search_res)
                        if not eval_result:  
                            refine_again = True
                            retry +=1
                            output = self.apply_retry_process(retry, response, user_input)
                            
                            messages.append({"role":"user", "content" : output})
                            response = self.agent.chat(
                                self.llama_model_id, 
                                messages=messages,
                                tools = [retrieve_acquisition_tool]
                                )
                            
                            print("refine_again : ",refine_again)
                            print("retry response ->",response.message.tool_calls)
                    elif  response.message.tool_calls is None and intent  in ["SEARCH", "DOC_WRITER"] and retry <2 and info_provided == False:
                        print(f"Even though the intent is {intent} the intent is  {response.message.tool_calls }. Retrying ...")
                        output = f"Your did not follow your instructions. Your intent is  '{intent}' so you should call the tool to process it."
                        retry +=1
                        messages.append({"role":"user", "content" : output})
                        response = self.agent.chat(
                            self.llama_model_id, 
                            messages=messages,
                            tools = [retrieve_acquisition_tool, create_doc_agents_tool]
                            )
                        
                        print("Retry response tools->",response.message.tool_calls)
                        
                print('Final response:', response.message.content)
                print("called_tools ->",called_tools)

            else:
                print("No tools triggered ...")

            if "get_intent_tool" not in called_tools and "retrieve_data_tool" not in called_tools:
                sources_update([])
            
            message_history.append(response.message)
            cl.user_session.set("message_history", message_history)

            return response.message.content.replace('"','') 
    
    @cl.step(type="tool", name = "Preparing the retry process ...")#,  language = user_language)
    def apply_retry_process(self, retry, response, user_input):
        refine_again = True
        retry +=1
        print("Answer needs review. Retry number ", retry)
        
        output = f"""The answer '{response.message.content}' is not relevant to the question '{user_input}'. 
        Analyze the gaps based on the feedback, refine the query and retrieve new information from the knowledge base by calling the tool 'retrieve_data_tool'.
        Answer the user's question based on these new search results and output the result in the user prefered language using the tool 'translate_output_tool'."""
        return   output

    def get_tools(self):
        return self.get_intent_olama_tool(), self.retrieve_data_olama_tool(), self.translate_output_olama_tool(), self.evaluate_responses_tool(), self.create_doc_agents_tool()
