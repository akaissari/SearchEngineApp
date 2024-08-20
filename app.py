import streamlit as st
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent,AgentType
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
import os 


GROG_API_KEY = st.secrets['GROQ_API_KEY']
GOOGLE_API_KEY =  st.secrets['GOOGLE_API_KEY']
GOOGLE_CSE_ID =  st.secrets['GOOGLE_CSE_ID']
LANGCHAIN_API_KEY = st.secrets['LANGCHAIN_API_KEY']
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["GOOGLE_CSE_ID"] = GOOGLE_CSE_ID
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "SearchEngineApp"



search = GoogleSearchAPIWrapper()

google_tool = Tool(
    name="google_search",
    description="Search Google for recent results.",
    func= lambda query: search.results(query, 5),
)


st.title("🔎 LangChain - Chat with search From Google")
"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

## Sidebar for settings
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your Groq API Key:",type="password")


if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assisstant","content":"Hi,I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

prompt = st.chat_input(placeholder="What is machine learning?")
if prompt:
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    llm=ChatGroq(groq_api_key=GROG_API_KEY,model_name="Llama3-70b-8192",streaming=True)

    tools=[google_tool]

    search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors=True)

    with st.chat_message("assistant"):
        #st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        #response=search_agent.run(st.session_state.messages,callbacks=[st_cb])
        response=search_agent.run(st.session_state.messages)
        st.session_state.messages.append({'role':'assistant',"content":response})
        st.write(response)

