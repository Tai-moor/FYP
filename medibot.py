import os
import streamlit as st
import csv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.tools import Tool, tool 
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv, find_dotenv

# Load the .env file
load_dotenv(find_dotenv())

# --- Configuration ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
# We use the 8b model for EVERYTHING to avoid rate limits
AGENT_MODEL = "llama-3.1-8b-instant" 
DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DOCTORS_CSV_PATH = "data/doctors.csv"
# ---------------------

# --- TOOL 1: Doctor Finder ---
@tool
def find_a_doctor(specialty: str, city: str) -> str:
    """
    Use this tool **if and only if** the user wants to find a doctor.
    This tool **REQUIRES** two arguments: `specialty` and `city`.
    If you don't have both from the `input` or `chat_history`, you **MUST** ask for the missing information.
    """
    if not specialty or not city:
        return "I am sorry, but I need both a medical specialty and a city to find a doctor."

    print(f"[Debug Tool]: Running find_a_doctor with specialty='{specialty}' and city='{city}'")
    results = []
    try:
        with open(DOCTORS_CSV_PATH, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if specialty.lower() in row['specialty'].lower() and city.lower() in row['city'].lower():
                    results.append(
                        f"Name: {row['name']}, Specialty: {row['specialty']}, City: {row['city']}, Address: {row['address']}, Phone: {row['phone']}"
                    )
        
        if not results:
            print("[Debug Tool]: No doctors found.")
            return f"No doctors were found for the specialty '{specialty}' in '{city}'."

        print(f"[Debug Tool]: Found {len(results)} doctors.")
        return "\n".join(results)

    except Exception as e:
        print(f"[Debug Tool]: Error reading CSV - {e}")
        return f"An error occurred while searching for doctors: {e}"

# --- TOOL 2: Symptom Checker (LLM VERSION) ---
@tool
def check_symptoms(symptoms: str) -> str:
    """
    Use this tool **if and only if** the user is describing their *own personal symptoms* (e.g., "I have a fever and cough", "my head hurts").
    """
    print(f"[Debug Tool]: Running LLM check_symptoms with symptoms='{symptoms}'")
    
    llm = ChatGroq(model_name=AGENT_MODEL, api_key=GROQ_API_KEY)
    
    symptom_prompt_template = """
**CRITICAL SAFETY RULE:** You are an AI, not a medical professional. You MUST start your response with this exact disclaimer:
"As an AI, I am not a medical professional. This is not a diagnosis. Please consult a real doctor for any health concerns."

**YOUR TASK:**
After the disclaimer, you **MUST ONLY** provide a bulleted list of *possible* associated medical conditions based on the user's symptoms.

**CRITICAL RULES:**
- **DO NOT** define the symptoms.
- **DO NOT** add any extra concluding paragraphs or advice.
- Your response **MUST** be *only* the disclaimer and the bulleted list.

**User's Symptoms:** "{user_symptoms}"

**Your Response:**
"""

    symptom_prompt = ChatPromptTemplate.from_template(symptom_prompt_template)
    
    symptom_chain = symptom_prompt | llm
    response = symptom_chain.invoke({"user_symptoms": symptoms})
    
    return response.content
# --- End of Tool 2 ---


# --- Load Resources (Cached) ---
@st.cache_resource
def get_vector_store():
    print("Loading vector store...")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    db = FAISS.load_local(
        DB_FAISS_PATH, 
        embedding_model, 
        allow_dangerous_deserialization=True
    )
    print("Vector store loaded.")
    return db

# --- Main Agent Setup ---
@st.cache_resource
def get_agent_executor():
    print("Creating Agent and Executor...")
    
    llm = ChatGroq(model_name=AGENT_MODEL, api_key=GROQ_API_KEY)
    
    # 2. Setup Tools
    
    # TOOL 1: The RAG Retriever
    retriever = get_vector_store().as_retriever(search_kwargs={'k': 3})
    rag_prompt = ChatPromptTemplate.from_template(
        """Answer the user's question based only on the context provided:
        Context: {context}
        Question: {input}
        Answer:"""
    )
    document_chain = create_stuff_documents_chain(llm, rag_prompt)
    retriever_chain = create_retrieval_chain(retriever, document_chain)

    def invoke_retriever_chain(query: str) -> dict:
        """
        Invokes the RAG chain. Expects a string query,
        converts it to the required dict, and returns the chain's result.
        """
        print(f"[Debug Tool]: Calling medical_book_search with query='{query}'")
        return retriever_chain.invoke({"input": query})

    retriever_tool = Tool(
        name="medical_book_search",
        func=invoke_retriever_chain,
        description="Use this tool ONLY for factual questions about medicine (e.g., 'What is dengue?')."
    )

    tools = [retriever_tool, find_a_doctor, check_symptoms]
    
    # 3. Create the Agent Prompt
    
    # --- **THIS IS THE FINAL, 10000% FIX** ---
    # This prompt is a simple "router". It does not allow the 8b model
    # to "think" or "summarize". It just forces it to pass the tool output.
    SYSTEM_PROMPT = """You are a simple router. Your only job is to choose the correct tool.

**#1 MOST IMPORTANT RULE: PASS-THROUGH**
After a tool provides an output, you **MUST** output that *exact* text.
Do not summarize it. Do not change it. Do not add any text. Just pass the tool's output.

**YOUR TOOL-CHOICE RULES:**

1.  **For User Symptoms (e.g., "I have a fever", "I feel sick"):**
    * You **MUST** call the `check_symptoms` tool.

2.  **For Finding a Doctor (e.g., "find a doctor in islamabad"):**
    * You **MUST** call the `find_a_doctor` tool.
    * (Check `chat_history` for `specialty` and `city`. If missing, you MUST ask for them.)

3.  **For Factual Questions (e.g., "What is dengue?"):**
    * You **MUST** call the `medical_book_search` tool.

4.  **For Non-Medical queries (e.g., "hi", "who is Messi"):**
    * You **MUST NOT** call any tool.
    * Your ONLY response is: "I am a medical assistant and can only answer medical-related questions. How can I help you with a medical query?"
"""
    # -----------------------------------------------

    agent_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # 4. Create the Agent
    agent = create_tool_calling_agent(llm, tools, agent_prompt)

    # 5. Create the Agent Executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    print("Agent and Executor created.")
    return agent_executor
# --- End of Agent Setup ---

# --- Streamlit App Main Function ---
def main():
    st.set_page_config(page_title="Ask Medibot Pro", page_icon="🧑‍⚕️")
    st.title("Ask Medibot Pro")
    st.markdown("Your AI assistant for medical info and doctor lookup.")

    try:
        agent_executor = get_agent_executor()
    except Exception as e:
        st.error(f"Failed to initialize the chatbot. Error: {e}")
        st.error(f"Debug: {e.__class__.__name__}, {e}")
        st.error("Please check your API keys and file paths.")
        return

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [] 
    
    # Display past chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get new user input
    if prompt := st.chat_input("Ask about a medical condition or find a doctor..."):
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Invoke the AGENT
                    response = agent_executor.invoke({
                        "input": prompt,
                        "chat_history": st.session_state.chat_history 
                    })
                    
                    answer = response.get("output", "I'm sorry, I couldn't process that.")
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    # Add to agent history
                    st.session_state.chat_history.append(HumanMessage(content=prompt))
                    st.session_state.chat_history.append(AIMessage(content=answer))
                    
                    st.markdown(answer)

                except Exception as e:
                    st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()