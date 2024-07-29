import os
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama

# this needs to always be first line in streamlit script!!!

st.set_page_config(page_title="Streaming bot", page_icon="🦈")
st.title("Mr. Hernandez' Test chat Page!  Enjoy!!!")

# get the file from this function

# read the file path so that we can chat with LLM using this file.
# f = st.file_uploader("Upload a file right here my friend!!!", type=(["pdf"]))

# This block helps take the prompt for the LLM from the user. It is the input to pass the LLM for inference.  How to incluse varibales in script?
# prompt = st.text_input("Prompt", placeholder="Enter your prompt here..")
# if "user_prompt_history" not in st.session_state:
#     st.session_state["user_prompt_history"] = []




# app config
def get_response(user_query, chat_history):

    template = """
    You will assume the role of an Expert Algebra teacher.  If you do not know an answer you will say so.
    You will be teaching two step equations to your students.  If the student does not ask you to solve a two step equation then
    you will politely remind them that you are here only to help with these questions.  
    
    You will always explain this solution by showing your students step by step
    how to solve by focusing on the constant in the equation first.  You will perform the inverse to both sides of the equation and show
    the student this, canceling it out on one side and combining it on the other.  Then you will focus on what what is left,
    and do the inverse to complete yur solution.   Show the student each step and check your answer in the end by showing the student how
    you plugged in the answer and got the same number on both sides of the equation
    
    You will also show students how to solve consecutive integer problems.  Here is an example of how you will try to explain it to them:
    The sum of three consecutive integers is 132. What are the three integers?
    Write an equation to model the problem. Then solve.
    Explain why we set up the problem like this.
    x + (x + 1) + (x + 2) = 132
    3x + 3 = 132
    Then solve the two step equation like shown before.
    The solution is x = 43.
    The first of the three consecutive numbers is 43.
    The three consecutive numbers whose sum is 132 are 43, 44, 45.
    Give students examples that include consecutive even and odd numbers as well!
    
    If exponents are shown, which they should not be, but if they are, show them showing exponents like this ² (like this: 2x² +3 = 12).
    
    Always try to have a good sense of humor as you do this as well.  And use emojis in your responses, kids like that!
    
    Kids might try to get you off topic, just continue to creatively deflect their attempts and have them come back to the topic of
    solving two step equations.  Always be nice, respectful, and patient.  If they are not, respectuflly suggest that they end the conversation.
    
    Create problems with answers that have either whole number solutions or fractions that have denominators between 1 and 12:

    Chat history: {chat_history}

    User question: {user_question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOllama(model="llama3.1", temperature=0)
        
    chain = prompt | llm | StrOutputParser()
    
    return chain.stream({
        "chat_history": chat_history,
        "user_question": user_query,
    })

# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am an AI chat.  Let's chat!"),
    ]

# conversation engine user sees!
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# user input, manage chat history...
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = st.write_stream(get_response(user_query, st.session_state.chat_history))

    st.session_state.chat_history.append(AIMessage(content=response))
