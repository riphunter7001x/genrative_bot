from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message
from utils import *

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Customizing Streamlit layout
st.set_page_config(
    page_title="Banking Assistant",
    page_icon=":bank:",
    layout="wide",
)

# Customizing UI colors
st.markdown(
    """
    <style>
        body {
            background-color: #e6f7ff;  /* Light Blue */
            color: #333;  /* Dark Gray */
            font-family: 'Arial', sans-serif;
        }
        .st-bw {
            background-color: #004080;  /* Navy Blue */
            color: white;
        }
        .st-title {
            color: #004080;
            text-align: center;
        }
        .st-subheader {
            color: #004080;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Banking Assistant")
st.subheader("Enhancing Customer Support with AI")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you today?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

system_msg_template = SystemMessagePromptTemplate.from_template(template="""You are a helpful assistant, you will use the provided context to answer user questions.
Read the given context before answering questions and think step by step. If you cannot answer a user question based on 
the provided context, inform the user. Do not use any other information for answering the user. Provide a detailed answer to the question.""")

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

# container for chat history
response_container = st.container()

# container for text box
textcontainer = st.container()

# Set the width of an empty placeholder to push the chat container to the right
st.markdown('<style>div.Widget.row-widget.stButton {width: 90%;}</style>', unsafe_allow_html=True)

with textcontainer:
    query = st.text_input("How can I help you today?", key="input")
    if query:
        with st.spinner("Thinking..."):
            conversation_string = get_conversation_string()
            refined_query = query_refiner(conversation_string, query)
            context = find_match(refined_query)
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)

# Move the response container to the right by setting its width
with response_container:
    st.markdown('<style>div.Widget.row-widget.stButton {width: 90%; margin-left: auto; margin-right: 0;}</style>', unsafe_allow_html=True)
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')
