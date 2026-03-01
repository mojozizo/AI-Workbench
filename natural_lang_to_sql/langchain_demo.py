# langchain_demo.py - Converted from langchain_demo.ipynb

import sqlite3
from langchain_community.utilities import SQLDatabase
from langchain.llms import Ollama
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import AgentType
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# --- Cell 1 & 2: Connect to SQLite database ---
db_path = "data/demo.db"
db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
print(db.dialect)
print(db.get_usable_table_names())
db.run("SELECT * FROM titanic LIMIT 10;")

# --- Cell 3: Table info (commented out) ---
# print(db.table_info)

# --- Cell 12 (placed before agent): Initialize LLM ---
llm = Ollama(model="sample")
print(llm.invoke("Who are you ?"))

# --- Cell 4: Print agent prompts ---
# Note: agent_executor not yet created here; moved after creation below
print("Setting up SQL agent...")

# --- Cell 5: Create SQL agent ---
system_prefix = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If the question does not seem related to the database, just return "I don't know" as the answer.
"""

system_suffix = """Frame the answer into a sentence and return it in an easy way to understand"""

agent_executor = create_sql_agent(
    llm,
    db=db,
    prefix=system_prefix,
    suffix=system_suffix,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# --- Print agent prompts (was Cell 4) ---
print(agent_executor.get_prompts)

# --- Cell 6: Commented out invocation ---
# agent_executor.invoke({"input": "Describe the titanic table"})

# --- Cell 7, 8, 9: Invoke agent ---
result = agent_executor.invoke("How many passengers were present in the Titanic ?")
result = agent_executor.invoke("Who are you ?")
result = agent_executor.invoke("Describe the titanic table ?")

# --- Cell 11: create_sql_query_chain + final chain ---
from langchain.chains import create_sql_query_chain

chain = create_sql_query_chain(llm, db)

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

execute_query = QuerySQLDataBaseTool(db=db)
answer = answer_prompt | llm | StrOutputParser()

final_chain = (
    RunnablePassthrough.assign(query=lambda x: x['sql_query']).assign(
        result=itemgetter("query") | execute_query
    )
    | answer
)

def ask_question(question):
    # Invoke the chain to get the SQL query
    response = chain.invoke({"question": question})
    # Extract the SQL query from the response
    sql_query = response.strip().split(': ')[2]
    # Get the final answer
    result = final_chain.invoke({"question": question, "sql_query": sql_query})
    return result
