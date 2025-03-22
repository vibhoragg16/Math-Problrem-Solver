import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, LLMMathChain
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_types import AgentType
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import Tool, initialize_agent


st.set_page_config(page_title="Text to Math Problem Solver And Data Search Assistant")
st.title("Text to Math Problem Solver Using google Gemma 2")

groq_api_key = st.sidebar.text_input("Enter your Groq API key", type="password")

if not groq_api_key:
    st.info("Please enter your Groq API key to continue")
    st.stop()

llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name = "Wikipedia",
    func = wikipedia_wrapper.run,
    description="A Tool for searching the internet to find variours information on the given topic"
)

math_chain = LLMMathChain.from_llm(llm = llm)
calculator = Tool(
    name = "Calculator",
    func=math_chain.run,
    description="A tool for answering math related questions. Omly input mathematical expressions"
)

prompt=""" 
Your are a Agent tasked for solving users mathematical question. Logically arrive at the solution and provide a detailed explaination
and display it point wise for the Question below.
Question: {question}
Answer: 
"""

prompt_template = PromptTemplate(
    input_variables=["question"],
    template=prompt
)
chain = LLMChain(llm=llm, prompt=prompt_template)

reasoning_tool = Tool(
    name = "Reasoning Tool",
    func = chain.run,
    description = "A tool for answering logic based and reasoning questions"
)

assistant_agent = initialize_agent(
    tools= [wikipedia_tool, calculator, reasoning_tool],
    llm = llm,
    agent= AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose= False,
    handle_parsing_errors= True
)

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role": "assistant", "content":"Hi, I'm a Math chatbot who can answer your maths questions"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


question = st.text_area("Enter your question","Amy bought a basket of fruits 1/5 of them were apples,1/4 were oranges, and the rest were 33 bananas. How many fruits did she buy in all?")

if st.button("find my answer"):
    if question:
        with st.spinner("Generate Response.."):
            st.session_state.messages.append({"role":"user", "content":question})
            st.chat_message("user").write(question)

            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = assistant_agent.run(st.session_state.messages, callbacks=[st_cb])
            st.session_state.messages.append({'role':'assistant','content':response})
            st.write('### Response:')
            st.success(response)

    else:
        st.warning("Please enter your input.")