# vana_ai_test.py - Converted from vana_ai_test.ipynb

from langchain_community.utilities import SQLDatabase
from langchain.llms import Ollama

# --- Cell 1 & 2: Connect to SQLite and initialize LLM ---
db_path = "data/sample.db"
db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
llm = Ollama(base_url="http://localhost:11434", model="project")
llm.invoke("Who are you")
print(db.get_usable_table_names())

# --- Cell 3: Set up Vanna with Ollama + ChromaDB ---
from vanna.ollama import Ollama
from vanna.chromadb import ChromaDB_VectorStore

class MyVanna(ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)

vn = MyVanna(config={'model': 'gemma:2b'})

# --- Cell 4: Connect Vanna to SQLite ---
vn.connect_to_sqlite('my-database.sqlite')

# --- Cell 5: Train from DDL ---
df_ddl = vn.run_sql("SELECT type, sql FROM sqlite_master WHERE sql is not null")
for ddl in df_ddl['sql'].to_list():
    vn.train(ddl=ddl)

# --- Cell 6: Train with custom DDL and documentation ---
vn.train(ddl="""
    CREATE TABLE IF NOT EXISTS my-table (
        id INT PRIMARY KEY,
        name VARCHAR(100),
        age INT
    )
""")

# Sometimes you may want to add documentation about your business terminology or definitions.
vn.train(documentation="Our university defines base stations as BS-XXX, where XXX is the base station number.")

# --- Cell 7: Get training data ---
training_data = vn.get_training_data()
print(training_data)

# --- Cell 8: Ask a question ---
vn.ask(question="How many base stations are there ?")
