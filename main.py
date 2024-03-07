from langchain_helper import get_few_shots_db_chain
import streamlit as st

st.title("👕The Tshirt Store!")

question = st.text_input("Question: ")

if question:
  chain = get_few_shots_db_chain()
  ans = chain.run(question)
  st.header("Answer: ")
  st.write(ans)