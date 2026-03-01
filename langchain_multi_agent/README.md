# Multi-Agent Architecture with LangChain

A demonstration of a multi-agent system architecture using LangChain with function calling, tool use, and Retrieval-Augmented Generation (RAG) capabilities. This project showcases how specialized agents can collaborate to solve complex tasks.

## System Overview

The architecture consists of four specialized agents, each with a specific duty:

- **Coordinator Agent**: Orchestrates the workflow and delegates tasks
- **Research Agent**: Performs RAG operations to retrieve relevant information
- **Analysis Agent**: Processes and analyzes information
- **Action Agent**: Executes specific tools and functions

## Architecture Explanation

### 1. Coordinator Agent

This agent serves as the "brain" of the system, responsible for:

- Understanding user requests
- Breaking down complex tasks into subtasks
- Delegating tasks to specialized agents
- Determining execution order
- Synthesizing results into a coherent final response

The coordinator uses a JSON output format to create a structured plan that specifies which agent should handle each subtask and in what order.

### 2. Research Agent (RAG Implementation)

This agent integrates Retrieval-Augmented Generation (RAG) capabilities:

- Uses FAISS vector store for document retrieval
- Converts documents to embeddings using Mistral AI embeddings
- Retrieves relevant context for a given query
- Provides information based on the retrieved knowledge
- Indicates when information is missing from its knowledge base

### 3. Analysis Agent

This agent focuses on processing and analyzing information:

- Takes information (often from the Research Agent)
- Identifies patterns and insights
- Draws conclusions
- Evaluates the significance of findings
- Provides structured analysis

### 4. Action Agent (Function & Tool Calling)

This agent performs concrete actions using available tools:

- Equipped with LangChain tools including:
  - Weather search tool
  - News search tool
  - Calculator tool
- Uses LangChain's agent executor to handle tool selection
- Can be expanded with additional tools as needed

## System Flow

1. User submits a query
2. Coordinator breaks it down and creates an execution plan
3. Tasks are executed by specialized agents in the specified order
4. Results from each agent are collected
5. Final response is synthesized and returned to the user

## Code Explanation

### Setup and Dependencies

The system requires:
- Mistral API key (stored in an environment variable)
- LangChain libraries for agents, tools, and embeddings
- FAISS for vector storage
- Text processing utilities

### Agent Implementation

1. **Tools Definition**:
   - `search_weather`: Simulates retrieving weather information
   - `search_news`: Simulates news search
   - `calculate`: Performs mathematical calculations

2. **RAG System**:
   - Loads and splits documents from a sample data file
   - Creates embeddings using Mistral AI
   - Builds a FAISS vector store for efficient retrieval

3. **Agent Definitions**:
   - Each agent has a specific prompt template defining its role
   - The Action Agent is built with LangChain's OpenAI tools agent framework
   - The Research Agent uses the RAG retriever to access knowledge

4. **Execution Flow**:
   - The `MultiAgentSystem` class orchestrates the entire process
   - It processes user queries through the Coordinator
   - Tasks are executed sequentially with built-in delays to avoid rate limiting
   - Results are synthesized into a coherent response

### Example Usage

The code includes examples demonstrating:
- Weather information retrieval and analysis
- Mathematical calculations with the calculator tool

## Getting Started

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up environment variables:
   ```
   MISTRAL_API_KEY=your_api_key_here
   ```

3. Run the example:
   ```
   python multi-agent_system.py
   ```

## Educational Value

This system effectively demonstrates:

- **Specialization**: Each agent has a clear, distinct role
- **Collaboration**: Agents work together on complex tasks
- **Tool Use**: Integration of external tools via function calling
- **RAG Capabilities**: Information retrieval and knowledge base integration
- **Orchestration**: Task delegation and execution management

## Extending the System

You can extend this system by:

- Adding more specialized agents
- Integrating additional tools
- Expanding the knowledge base
- Implementing agent-to-agent communication
- Adding more sophisticated memory systems for contextual awareness