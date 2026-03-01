# chatollama.py - Converted from chatollama.ipynb

from langchain.utilities import SQLDatabase
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# --- Cell 1: Connect to SQLite database ---
connection_string = 'sqlite:///data/sample.db'
db = SQLDatabase.from_uri(connection_string)
print(db.dialect)
print(db.get_usable_table_names())

# --- Cell 2: Get database context ---
context = db.get_context()
print(list(context))
print(context["table_info"])

# --- Cell 3: Initialize ChatOllama model ---
chat_model = ChatOllama(
    model="deepseek-r1:1.5b",
    temperature=0.7,
    base_url="http://localhost:11434",
)

response = chat_model.invoke("Explain what SQL injection is in one sentence.")
print("Basic response:\n", response)

# --- Cell 4: Chat with system + human messages ---
messages = [
    SystemMessage(content='''You are a SQLite expert. Given an input question, first create a syntactically correct SQLite query to run, then look at the results of the query and return the answer to the 
input question.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Pay attention to use date('now') function to get the current date, if the question involves "today".
If mot mentioned do not use {top_k} rows. Output only the SQL Query without any output or result of the query.
If the answer could not be found, give a single line meaningful error'''),
    HumanMessage(content="Count the number of base stations?")
]

response = chat_model.invoke(messages)
print("Response with context:\n", response)
