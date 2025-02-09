# GenAI - Virtual Assistant 

## Overview

This Virtual Assistant is designed to provide intelligent responses to both general conversational inputs and domain-specific queries related to scientific research. Additionally, it has the capability to generate structured documents following academic paper conventions using a multi-agent system.

## Features

### 1. Conversational Capabilities

The assistant can handle general conversational inputs such as:

"Hello!"

"How are you?"

"Tell me a joke."

Provides a smooth and natural conversational experience.

### 2. Research-Specific Question Answering

Uses Retrieval-Augmented Generation (RAG) to answer questions about:

- Thesis

- Research papers

- Scientific articles

Any question about a specific topic will trigger this feature.

Ensures accurate and contextually relevant responses by relying solely on retrieved information.

### 3. Structured Document Generation

The assistant can generate a research-style document with the following sections :

- Title, Introduction, Abstract

- Body of the document

- Conclusion and References

#### Multi-Agent Implementation Approaches

The multi-agent system delegates different sections of the document to specialized agents, while a supervisor is used to coordinate the multiple agents tasks (Fig. 1).

![Multi-Agents Diagram  ](Virtual_Assistant/images/supervisor-diagram.png)
Fig. 1 - Multi-Agents Diagram  [https://blog.langchain.dev/langgraph-multi-agent-workflows/]


The multi-agent system is implemented in two different ways:

1. LangGraph with Ollama (Llama 3.2)

Leverages LangGraph to create a structured workflow for multi-agent interactions, implementing a Supervisor-Worker model.

Uses Ollama and Llama 3.2 as the underlying language model.

Ensures seamless communication between the agents handling different sections, while the supervisor coordinates the workflow.

2. Custom Multi-Agent Implementation

Implements a Supervisor-Worker model using Ollama (Llama 3.2), with the Supervisor coordinating the workflow and deciding which worker agent to work next.


**Three Worker Agents handle specific sections:**

- Agent 1 : Title, Introduction, and Abstract

- Agent 2 : Body of the document

- Agent 3 : Conclusion and References

Provides greater flexibility and control over agent interactions, and increase the tokens limits while generating the document.

## Technology Stack

- Ollama (Llama 3.2) – Core language model for text generation.

- LangGraph – Graph-based framework for managing multi-agent workflows.

- RAG (Retrieval-Augmented Generation) – For answering scientific queries.

- Python 3

- Chainlit

-  PostgreSQL + pgvector 

# Install dependencies
pip install -r Virtual_Assistant/chainlit_app/requirements.txt

# Run the assistant
chainlit run app.py -w