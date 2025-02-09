import chainlit as cl
import os
from llama_index.core.postprocessor import LLMRerank
from llama_index.core import QueryBundle
from llama_index.core.base.llms.types import ChatMessage

llama_model_id = os.getenv("llama_model_id", None)
#=== Get chunks from bd based on embedding similarity

def retrieve_chunks(user_input):

    # Generate embedings for query similirity in the knowledge base
    query_result_list = cl.user_session.get("retriever").retrieve(user_input)

    if len(query_result_list)> 10:

        # choice_batch_size parameter is used to specify the number of items processed at a time during reranking. 
        # When you have a large number of items to rerank, batching can improve efficiency by reducing memory usage or managing API call limits.

        # How it works:
        # Purpose: Divides the input data into smaller chunks (batches) for reranking. Each batch is processed independently before the results are combined.
        # Default Value: If not specified, the default value varies based on the implementation, but it's typically optimized for performance.
        # Use Case: You can set this to a smaller number if you're facing memory constraints or to a higher value to improve throughput.


        # configure reranker
        reranker = LLMRerank(
            choice_batch_size=5,
            top_n=10
        )
        query_result_list = reranker.postprocess_nodes(
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
                doc_name = metadata["file_path"].split("/")[-1].replace(".pdf","") + " - Page "+metadata["page_label"]
                sources_elements.append(cl.Pdf(name=doc_name, path = metadata["file_path"], display="inline") )

    sources_update(sources_elements)

    return sources

def get_intent(user_input: str) -> str:
    """Getting the intent of the user's input to understand if the user intention is to search 
    for specific information from the knowledge base or just having a conversational/normal interaction.
    This should be the first step of the agent."""

    final_query  = f"""
    You are an AI assistant specialized in understanding the user's intent.
    Given the user input, output:
    - Output "SEARCH" if the user's intention is to search for specific information such as works/thesis/papers, authors/persons, etc,  or related details.
    - Output "NORMAL" otherwise.

    Do not explain your reasoning. Just output "SEARCH" or "NORMAL" based on the intent of the input.
    """
    messages = [
        ChatMessage(
            role="system", content=final_query
        ),
        ChatMessage(role="user", content=user_input),
    ]
    
    #Extract the retrieved context and sources
    resp = cl.user_session.get("model").chat(messages, max_tokens=200, temperature = 0)

    if resp.message.blocks[0].text == "NORMAL":
        sources_update([])

    return resp.message.blocks[0].text

def search_intent(model, user_input):
    final_query  = f"""
    You are an AI assistant specialized in understanding the user's intent.
    Given the user input, output:
    - Output "1" if the user's query indicates an intention to search for specific information from the knowledge base.
    - Output "0" if the user's query is conversational or not related to searching for specific information.

    Do not explain your reasoning. Just output "1" or "0" based on the nature of the input.
    Input: {user_input}
    """
    messages = [
        ChatMessage(
            role="system", content=final_query
        ),
        ChatMessage(role="user", content=user_input),
    ]
   
    #Extract the retrieved context and sources
    resp = model.chat(messages, max_tokens=2000, temperature = cl.user_session.get("temperature_param"))

    return int(resp.message.blocks[0].text)

def retrieve_and_generate(agent, history, user_input, context = None ):
    if len(context) > 0:
        system_instruction = [
                            "You are a chatbot, able to have normal interactions, as well as talk about information Papers and Thesises.",
                            "If you are not sure or don't know the answer, say you don't know and ask if there is something else you can help the user with.",
                            f"Here you have the search result: {context}."
                        ]

        if len(history)> 0 : 
            system_instruction.append(f"Here you have the conversation history: {history}")
            system_instruction.append(f"Based on the conversation history and the search result, please answer the user's input")
        else:
            system_instruction.append(f"Based on search result please answer the user's input")
        
        system_instruction.append(f"If no answer is found based on the search result and the conversation history, try to answer using your own knowledge.")
    else:
        system_instruction = [
                            "You are a chatbot, able to have normal interactions, as well as talk about information Papers and Thesises.",
                            "Provide a user friendly answer to the user's input.",
                        ]

    system_instruction.append(f"You should ALWAYS answer using the language {cl.user_session.get('language')} .")


    final_query = '\n'.join(system_instruction)
    print("final_query ->",final_query)

    messages = [
        ChatMessage(
            role="system", content=final_query
        ),
        ChatMessage(role="user", content=user_input),
    ]

    #Extract the retrieved context and sources
    resp = agent.chat(messages, max_tokens=4000, temperature = cl.user_session.get("temperature_param"))
    return resp.message.blocks[0].text

#=== Receives a List of chainlit elements and appends it
def sources_update(elements):
    answer_sources = cl.user_session.get("answer_sources")
    answer_sources.append(elements)
    cl.user_session.set("answer_sources", answer_sources)


#=== Receives a Chainlit Message and appends it
def update_conversation_history(msg):

    conversation_history = cl.user_session.get("conversation_history")
    conversation_history.append(msg)
    cl.user_session.set("conversation_history", conversation_history)

    resume_history = cl.user_session.get("resume_history")
    resume_history.append(msg.content)
    cl.user_session.set("resume_history", resume_history)

