import chainlit as cl
import copy
import re
import json
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

members = [
    "intro_creator",
    "body_creator",
    ]

system_prompt = (
    f"You are a supervisor tasked with managing a conversation between the following workers: {members}. "
    "Each worker is specifialized in writing a different parts of documents and it is not expected to write more then its own sections :"
    "The workers should called in the following order : intro_creator, body_creator." 
    "Given the following user request, respond with the worker to act next. "
    "Each worker will perform atask and respond with their results and status. "
    "When finished, respond with FINISH."
)


class State(MessagesState):
    next: str

class AIAgent():
    def __init__(
            self,
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
        self.new_members = copy.deepcopy(members)
        self.new_system_prompt = copy.deepcopy(system_prompt)
        self.llm = ChatOllama(model=supervisor_llm, temperature=0.9, num_predict  = 8000)#, repeat_penalty = 1.2)
        self.workers = ChatOllama(model=worker_llm, temperature=0.9, num_predict  = 8000)
        self.retriever = retriever

        self.body_creator_agent = create_react_agent(self.workers, tools=[], state_modifier="You are an agent specialized in writing the body, the conclusion and the reference sections of a document. Provide the maximum information as possible. You do not write anything apart from the refered sections .")
        self.intro_creator_agent = create_react_agent(self.workers, tools=[], state_modifier="You are an agent specialized in writing the introduction, the abstract and the state of the art sections of a document. Provide the maximum information as possible. You do not write anything apart from the refered sections .")

        
        self.builder = StateGraph(State)
        self.builder.add_edge(START, "supervisor")
        self.builder.add_node("supervisor", self.supervisor_node)
        self.builder.add_node("intro_creator", self.intro_creator_node)
        self.builder.add_node("body_creator", self.body_creator_node)
        self.worker_finished = []
        self.running_worker = 0
        self.graph = self.builder.compile()

    #https://stackoverflow.com/questions/78404535/langchain-python-with-structured-output-ollama-functions

    def supervisor_node(self,state: State) -> Command[Literal[*members, "__end__"]]:
        print("\n ------------------------------------\nsupervisor input state : ", state, "\n ------------------------------------\n")
        
        if "next" in state:
            if state["next"] in self.new_members:
                self.new_members.remove(state["next"])

                print(f"Updating prompt with available workers '{self.new_members}' ... ")
                self.new_system_prompt = (
                    f"You are a supervisor tasked with managing a conversation between the following workers: {self.new_members}. "
                    "Each worker is specifialized in writing a different parts of documents and it is not expected to write more then its own sections."
                    "Given the following user request, respond with the worker to act next focusing ONLY in the the specified workers. Each worker will perform a"
                    " task and respond with their results and status. When finished,"
                    " respond with FINISH."
                )
    
        options = self.new_members + ["FINISH"]
        print(f"Updating options with available workers '{options}' ... ")

        class Router(TypedDict):
            """Worker to route to next. If no workers needed, route to FINISH."""
            next: Literal[*options]
        
        while True: # To deal wil the ChatOllama bug that somethings return just None instead of a structured output (even defining the output format as json, for example)
            if len(self.new_members) == 0:
                goto = END
                break
            else:
                try:
                    messages = [{"role": "system", "content": self.new_system_prompt},] + state["messages"]
                    response = self.llm.with_structured_output(Router).invoke(messages)
                    print("supervisor node response : ",response)
                    goto = response["next"]
                    
                    if goto not in self.worker_finished:
                        self.worker_finished.append(goto)

                    if goto == "FINISH" or len(self.new_members) == 0:
                        goto = END
                    
                    if goto not in self.new_members :
                        goto = "supervisor"
                    
                    break

                except Exception as e:
                    print("Something went wrong in the supervisor. Ending the multi-agent interaction. \nResponse : ",response, "\nError : ", e)#," \nmessages",messages)
                    goto = "supervisor"

        if goto == END :
            self.new_members = copy.deepcopy(members)

            self.new_system_prompt = (
                "You are a supervisor tasked with managing a conversation between the"
                f" following workers: {self.new_members}. Given the following user request,"
                " respond with the worker to act next.Do not repeat the same worker consecutively. Each worker will perform a"
                " task and respond with their results and status. When finished,"
                " respond with FINISH."
            )

        return Command(goto=goto, update={"next": goto})

    def intro_creator_node(self, state: State) -> Command[Literal["supervisor"]]:
        result = self.intro_creator_agent.invoke(state)

        print("Introduction creator output : ",result["messages"][-1].content)
        return Command(
            update={
                "messages": [
                    HumanMessage(content=result["messages"][-1].content, name="intro_creator")
                ]
            },
            goto="supervisor",
        )


    def body_creator_node(self, state: State) -> Command[Literal["supervisor"]]:
        result = self.body_creator_agent.invoke(state)

        print("Body creator output : ",result["messages"][-1].content)
        return Command(
            update={
                "messages": [
                    HumanMessage(content=result["messages"][-1].content, name="body_creator")
                ]
            },
            goto="supervisor",
        )

    async def retrieve_chunks(self, user_input):

        query_result_list = await self.retriever.aretrieve(user_input)

        #print(query_result_list)
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

        resp = ollama.generate(model=self.llama_model_id, prompt=copy.deepcopy(INTENT_PROMPT).replace("$USER_INPUT$", user_input).replace("$HISTORY$", '\n'.join(cl.user_session.get("most_recent_history"))))
        if resp['response'] !="SEARCH" or resp['response'].find('SEARCH') == -1:  
            print("Filling empty sources as this is a question for not using RAG...")
            sources_update([])

        print("Intent detected : ", resp['response'])
        return resp['response']

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

    async def retrieve_data_from_kb(self, refine_again, refined_query, queries_done, user_input,):
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

        return queries_done, search_res, output, refined_query
    
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
            
            if isinstance(response, str):
                try:
                    response = json.loads(response)
                except json.JSONDecodeError:
                    print("Error decoding JSON response:", response)
                    response = {"message": {
                                        "content" : response,
                                        "tool_calls" : []
                                    }}

            called_tools = []
            intent = None
            tools_list = None
            refined_query = None
            search_res = None
            print("First model response ->",response.message)
            refine_again = False
            retry = 0
            queries_done = [user_input]
            cl_msg = None
            extracted_texts = []
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
                            tools_list = [language_tool ]

                        elif intent == "NORMAL" or intent.find('NORMAL') != -1:
                            output = f"The intent is '{intent}'. There's no need to search in the knowledge base. Just engage in a conversation with the user and answer politely to the user's input."
                            tools_list = [language_tool ]
                        

                    elif tool.function.name =="create_doc_tool":
                        answer = []

                        try:
                            topic_to_process = tool.function.arguments["topic"]
                        except : 
                            topic_to_process = user_input

                        print("Creating document tool : ", topic_to_process)
                        
                        for s in self.graph.stream( {"messages": [("user", topic_to_process)]}, { "recursion_limit": 50}, subgraphs=True ):
                            answer.append(s[1])

                                                
                        extracted_texts = []
                        seen_agents = set()
                        for entry in answer:
                            for agent, data in entry.items():
                                if agent not in seen_agents and "messages" in data:
                                    
                                    first_message = data["messages"][0].content if data["messages"] else ""
                                    
                                    if first_message in extracted_texts:
                                        pass
                                    else:
                                        extracted_texts.append(first_message)
                                        seen_agents.add(agent)
    
                        
                        extracted_texts = '\n\n'.join(extracted_texts)
                        print("Document created  : ",extracted_texts)
                        return extracted_texts
                        
                    elif tool.function.name == "retrieve_data_tool":
                        print("Retriving data from the knowledge base ...")

                        queries_done, search_res, output, refined_query =  await self.retrieve_data_from_kb( refine_again, refined_query, queries_done, user_input)
                        tools_list = [language_tool,  create_doc_agents_tool]

                    elif tool.function.name == "translate_output_tool":
                        print("Translating output...")
                        output = self.translate_output_tool(tool.function.arguments)
                        tools_list = None
                    

                    print("Calling the agent again")
                    messages.append(response.message)
                    messages.append({'role': 'tool', 'content': str(output), 'name': tool.function.name})

                    # Get final response from model with function outputs
                    print("Message created. Calling the agent with tool :", tools_list)
                    if tools_list is None:
                    
                        response = self.agent.chat(
                            self.llama_model_id, 
                            messages=messages,

                            )
                    else:
                        response = self.agent.chat(
                            self.llama_model_id, 
                            messages=messages,
                            tools = tools_list
                            )
                    
                    if isinstance(response, str):
                        try:
                            response = json.loads(response)
                        except json.JSONDecodeError:
                            print("Error decoding JSON response:", response)
                            response = {"message": {
                                        "content" : response,
                                        "tool_calls" : []
                                    }}

                    print('Response:', response.message.content)
                    print("Response tools : ",response.message.tool_calls)
                    print("Retry : ",retry)
                    print("refine_again : ",refine_again)
                    if not response.message.tool_calls and intent.find('SEARCH') != -1 and search_res is None:
                        print("SEARCH intent but no retrieve performed ...")
                        queries_done, search_res, output, refined_query =  await self.retrieve_data_from_kb( refine_again, refined_query, queries_done, user_input)
                        tools_list = [language_tool,  create_doc_agents_tool]
                        messages.append({"role":"user", "content" : output})

                        response = self.agent.chat(
                            self.llama_model_id, 
                            messages=messages,
                            tools = tools_list
                            )
                        
                        if isinstance(response, str):
                            try:
                                response = json.loads(response)
                            except json.JSONDecodeError:
                                print("Error decoding JSON response:", response)
                                response = {"message": {
                                        "content" : response,
                                        "tool_calls" : []
                                    }}
                                
                    if (not response.message.tool_calls and retry == 1 and search_res is not None) :
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

                            if isinstance(response, str):
                                try:
                                    response = json.loads(response)
                                except json.JSONDecodeError:
                                    print("Error decoding JSON response:", response)
                                    response = {"message": {
                                        "content" : response,
                                        "tool_calls" : []
                                    }}

                            print("retry response ->",response.message.tool_calls)
                
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
