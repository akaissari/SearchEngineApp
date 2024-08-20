import streamlit as st
from langchain_groq import ChatGroq
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import initialize_agent,AgentType
from langchain_openai import ChatOpenAI

import os

# Load secrets
GROG_API_KEY = st.secrets['GROQ_API_KEY']
OPEN_AI_KEY = st.secrets['OPENAI_AI_KEY']
GOOGLE_API_KEY = st.secrets['GOOGLE_API_KEY']
GOOGLE_CSE_ID = st.secrets['GOOGLE_CSE_ID']
LANGCHAIN_API_KEY = st.secrets['LANGCHAIN_API_KEY']

# Set environment variables
os.environ['OPENAI_API_KEY']=OPEN_AI_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["GOOGLE_CSE_ID"] = GOOGLE_CSE_ID
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "SearchEngineApp"

# Initialize search tool
search = GoogleSearchAPIWrapper()
google_tool = Tool(
    name="google_search",
    description="Search Google for recent results.",
    func=lambda query: search.results(query, 1),
)

st.title("🔎Chat with search From Google")

# Initialize conversation history in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        AIMessage(content="Hi, I'm a chatbot who can search the web. How can I help you?")
    ]

# Display the conversation history in the chat interface
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# Get user input
prompt = st.chat_input(placeholder="What is machine learning?")

if prompt:
    # Append user message to conversation history
    user_message = HumanMessage(content=prompt)
    st.session_state.messages.append(user_message)
    st.chat_message("user").write(prompt)

    # Initialize the language model (LLM)
    #llm = ChatGroq(groq_api_key=GROG_API_KEY, model_name="Llama3-70b-8192", streaming=True)
    llm=ChatOpenAI(model="gpt-4o")
    search = GoogleSearchAPIWrapper()
    google_tool = Tool(
    name="google_search",
    description="Search Google for recent results.",
    func= lambda query: search.results(query, 1),
    )
    tools=[google_tool]
    search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_errors=True)


    # Use the LLM to generate a response based on the conversation history
    with st.chat_message("assistant"):
        try:
            response = search_agent.invoke(st.session_state.messages)
            assistant_message = AIMessage(content=response['output'])
            st.session_state.messages.append(assistant_message)
            st.write(response['output'])
        except Exception as e:
            st.write(str(e))
