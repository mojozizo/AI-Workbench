# multi_agent_system.py
"""
A simple multi-agent architecture with function calling, tool use, and RAG capabilities.
This system demonstrates agents with specialized roles working together to accomplish tasks.
"""

import os
from typing import List, Dict, Any, Optional
import json
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_mistralai import ChatMistralAI
from langchain_core.tools import tool, StructuredTool
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.agents import AgentExecutor, create_openai_functions_agent, create_openai_tools_agent
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory

# Get API key from environment
mistral_api_key = os.getenv("MISTRAL_API_KEY")
if not mistral_api_key:
    raise ValueError("MISTRAL_API_KEY not found in environment variables")

# Initialize the LLM
llm = ChatMistralAI(model="mistral-medium", temperature=0, api_key=mistral_api_key)

# Create tools for agents to use
@tool
def search_weather(location: str) -> str:
    """Search for current weather in a given location"""
    # In a real application, this would call a weather API
    return f"The weather in {location} is currently sunny with a temperature of 75°F."

@tool
def search_news(query: str) -> str:
    """Search for recent news on a specific topic"""
    # In a real application, this would call a news API
    return f"Latest news on {query}: New developments have been reported recently."

@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"

# Define all available tools
tools = [search_weather, search_news, calculate]

# Set up RAG system
def setup_rag_system(documents_path="./knowledge_base"):
    """Set up the RAG system with the provided documents"""
    # Load documents
    loader = TextLoader("./sample_data.txt")  # Sample data file
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    
    # Create vector store
    embeddings = MistralAIEmbeddings(api_key=mistral_api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()
    
    return retriever

# Sample data for testing - in a real implementation, this would be an actual file
sample_data = """
Artificial Intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans.
Leading AI textbooks define the field as the study of "intelligent agents": any system that perceives its environment and takes actions that maximize its chance of achieving its goals.
Some popular AI methods include machine learning, deep learning, and reinforcement learning.

Multi-agent systems consist of multiple interacting intelligent agents. These systems can be used to solve problems that are difficult or impossible for an individual agent.
Agents can have different roles, capabilities, and knowledge, and they can work together to achieve a common goal.
"""

# Write sample data to a file
with open("sample_data.txt", "w") as f:
    f.write(sample_data)

# Create the RAG retriever
retriever = setup_rag_system()

# 1. Coordinator Agent
coordinator_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are the Coordinator Agent in a multi-agent system. Your role is to:
    1. Understand the user's request
    2. Break down complex tasks into subtasks
    3. Determine which agent should handle each subtask
    4. Synthesize results from different agents
    
    Available agents:
    - Research Agent: For retrieving and providing relevant information (RAG)
    - Analysis Agent: For analyzing and processing information
    - Action Agent: For executing specific tools and functions
    
    Output your plan as JSON with the following structure:
    {{
        "subtasks": [
            {{
                "id": "task1",
                "description": "Description of the subtask",
                "assigned_agent": "Research/Analysis/Action",
                "requires_tool": true/false,
                "tool_name": "tool_name_if_applicable"
            }}
        ],
        "execution_order": ["task1", "task2", ...]
    }}
    """),
    ("human", "{input}"),
])

coordinator_parser = JsonOutputParser()
coordinator_chain = coordinator_prompt | llm | coordinator_parser

# 2. Research Agent with RAG
research_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are the Research Agent in a multi-agent system. Your role is to research and provide relevant information.
    You have access to a knowledge base through a retrieval system. Use the retrieved context to answer questions thoroughly.
    If the retrieved information doesn't contain what you need, clearly state what information is missing.
    
    Respond with detailed, factual information based on the context provided.
    """),
    ("human", "Task: {task}\n\nRelevant information from knowledge base: {retrieved_info}")
])

def research_agent_executor(task: str):
    """Execute the research agent to retrieve and provide information"""
    # Retrieve relevant documents
    docs = retriever.invoke(task)
    retrieved_info = "\n\n".join([doc.page_content for doc in docs])
    
    # Process with the research agent
    response = llm.invoke(research_prompt.format(
        task=task,
        retrieved_info=retrieved_info
    ))
    
    return response.content

# 3. Analysis Agent
analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are the Analysis Agent in a multi-agent system. Your role is to analyze information and provide insights.
    You should:
    1. Process the information provided
    2. Draw connections and identify patterns
    3. Evaluate the significance of the information
    4. Provide clear conclusions
    
    Your analysis should be thorough, logical, and insightful.
    """),
    ("human", "Task: {task}\n\nInformation to analyze: {information}")
])

# Another approach to doing things, with chain you can add more steps like parsing or validation 
analysis_chain = analysis_prompt | llm

def analysis_agent_executor(task: str, information: str):
    """Execute the analysis agent to analyze information"""
    response = analysis_chain.invoke({
        "task": task,
        "information": information
    })
    
    return response.content

# 4. Action Agent with Tools
action_agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are the Action Agent in a multi-agent system. Your role is to execute specific actions using available tools.
    You have access to tools for retrieving weather data, searching news, and performing calculations.
    Use these tools appropriately to complete the assigned task.
    
    Think carefully about which tool is appropriate for the given task.
    """),
    ("human", "{input}"),
    ("human", "{agent_scratchpad}")
])

action_agent = create_openai_tools_agent(
    llm=llm,
    prompt=action_agent_prompt,
    tools=tools
)

action_agent_executor = AgentExecutor(agent=action_agent, tools=tools, verbose=True)

# Main system coordinator
class MultiAgentSystem:
    def __init__(self):
        self.memory = ConversationBufferMemory(return_messages=True)
    
    def process_query(self, user_query: str) -> str:
        """Process a user query using the multi-agent system"""
        # Step 1: Coordinator creates a plan
        try:
            plan = coordinator_chain.invoke({"input": user_query})
            print(f"Execution plan: {json.dumps(plan, indent=2)}")
            
            # Step 2: Execute the plan in the specified order
            results = {}
            for task_id in plan["execution_order"]:
                task = next(t for t in plan["subtasks"] if t["id"] == task_id)
                
                # Add delay to avoid rate limiting
                time.sleep(2)
                
                if task["assigned_agent"] == "Research":
                    results[task_id] = research_agent_executor(task["description"])
                    
                elif task["assigned_agent"] == "Analysis":
                    # Find the dependent task results if needed
                    information = results.get(task_id.replace("analyze", "research"), user_query)
                    results[task_id] = analysis_agent_executor(task["description"], information)
                    
                elif task["assigned_agent"] == "Action":
                    if task.get("requires_tool", False):
                        tool_input = f"Use the {task.get('tool_name', '')} tool to complete this task: {task['description']}"
                        results[task_id] = action_agent_executor.invoke({"input": tool_input})["output"]
                    else:
                        results[task_id] = "No specific tool required for this action."
                
                # Add delay after each task
                time.sleep(2)
            
            # Add delay before final synthesis
            time.sleep(2)
            
            # Step 3: Synthesize the results
            synthesis_prompt_template = ChatPromptTemplate.from_messages([
                ("system", "Synthesize the results from multiple agents into a coherent response."),
                ("human", "User query: {user_query}\n\nResults from agents: {results}")
            ])
            
            final_response = llm.invoke(synthesis_prompt_template.format(
                user_query=user_query,
                results=json.dumps(results, indent=2)
            ))
            return final_response.content
            
        except Exception as e:
            return f"Error in multi-agent system: {str(e)}"

# Example usage
def main():
    system = MultiAgentSystem()
    
    # Example query
    query = "What's the weather in New York and how does it relate to climate change?"
    print("\nUser Query:", query)
    response = system.process_query(query)
    print("\nFinal Response:", response)
    
    query = "Calculate the compound interest on $1000 with 5% annual interest over 5 years"
    print("\nUser Query:", query)
    response = system.process_query(query)
    print("\nFinal Response:", response)

if __name__ == "__main__":
    main()