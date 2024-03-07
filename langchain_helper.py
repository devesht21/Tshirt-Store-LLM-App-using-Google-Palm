from langchain.llms import GooglePalm
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from langchain.prompts.prompt import PromptTemplate

import config
from few_shots import few_shots

def get_few_shots_db_chain():

  llm = GooglePalm(google_api_key=config.GOOGLE_API_KEY, temperature=0.1)

  db = SQLDatabase.from_uri(f"mysql+pymysql://{config.db_user}:{config.db_password}@{config.db_host}/{config.db_name}",sample_rows_in_table_info=3)

  embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

  to_vectorize = [' '.join(example.values()) for example in few_shots]

  vector_stores = Chroma.from_texts(to_vectorize, embeddings, metadatas=few_shots)

  example_selector = SemanticSimilarityExampleSelector(vectorstore=vector_stores, k=2,)

  prompt_template = PromptTemplate(input_variables=["Question", "SQLQuery", "SQLResult", "Answer"], template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}")

  few_shot_prompt = FewShotPromptTemplate(example_selector=example_selector, example_prompt=prompt_template, prefix=_mysql_prompt, suffix=PROMPT_SUFFIX, input_variables=["input", "table_info", "top_k"])

  chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, prompt=few_shot_prompt)

  return chain







