import streamlit as st
import os
from langchain.chains.question_answering.map_reduce_prompt import system_template
from langchain.chains.summarize.refine_prompts import prompt_template
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
llm = ChatMistralAI(api_key=api_key)

# summarization part
def summarize_text(text_input):

    system_template = "Generate a concise summary of the following text."

    prompt_template = ChatPromptTemplate.from_messages(
        [("system",system_template), ("user" ,"{text}")]
    )

    prompt = prompt_template.invoke({"text": text_input})
    response = llm.invoke(prompt).content
    return response



# Tittle of the page
def main():
    st.title("Summary Generator")
    st.subheader("Input \n")
    text_input = st.text_area("Enter paragraph",max_chars=10000,height=350,value="",key="textbox")

    sum = st.button("Summary")

    if sum:
        result = summarize_text(text_input)
        st.text_area("Here is your summary:",value=result,height=350)

if __name__ == "__main__":
    main()
