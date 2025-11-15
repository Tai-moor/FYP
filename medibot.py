import os
import streamlit as st
import csv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.tools import Tool, tool 
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv, find_dotenv

# --- UPDATED IMPORTS ---
from langchain_google_genai import ChatGoogleGenerativeAI
# -------------------------------------

# Load the .env file
load_dotenv(find_dotenv())

# --- Configuration ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") 
AGENT_MODEL = "gemini-2.0-flash"
    # The "brain" for ALL tasks
DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DOCTORS_CSV_PATH = "data/doctors.csv"
# ---------------------

# --- Gemini Safety Settings ---
GEMINI_SAFETY_SETTINGS = {
    0: 0,  # HARASSMENT
    1: 0,  # HATE_SPEECH
    2: 0,  # SEXUALLY_EXPLICIT
    3: 0,  # DANGEROUS_CONTENT
}

# ------------------------------

# --- TOOL 1: Doctor Finder ---
@tool(description="Find a doctor by specialty and city.")
def find_a_doctor(specialty: str, city: str) -> str:
    """Returns a list of doctors matching the specialty and city."""
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

# --- TOOL 2: Symptom Checker ---
@tool(description="Check symptoms and provide possible medical conditions.")
def check_symptoms(symptoms: str) -> str:
    """Uses AI to suggest possible conditions based on symptoms."""
    print(f"[Debug Tool]: Running check_symptoms with symptoms='{symptoms}'")
    
    llm = ChatGoogleGenerativeAI(
        model=AGENT_MODEL, 
        google_api_key=GOOGLE_API_KEY,
        safety_settings=GEMINI_SAFETY_SETTINGS
    )
    
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

# --- Load Vector Store (Cached) ---
@st.cache_resource
def get_vector_store():
    print("Loading vector store...")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    print("Vector store loaded.")
    return db

# --- Main Agent Setup ---
@st.cache_resource
def get_agent_executor():
    print("Creating Agent and Executor...")

    llm = ChatGoogleGenerativeAI(
        model=AGENT_MODEL,
        google_api_key=GOOGLE_API_KEY,
        safety_settings=GEMINI_SAFETY_SETTINGS
    )

    # Setup Tools
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
        print(f"[Debug Tool]: Calling medical_book_search with query='{query}'")
        return retriever_chain.invoke({"input": query})

    retriever_tool = Tool(
        name="medical_book_search",
        func=invoke_retriever_chain,
        description="Use this tool ONLY for factual questions about medicine (e.g., 'What is dengue?')."
    )

    tools = [retriever_tool, find_a_doctor, check_symptoms]

    SYSTEM_PROMPT = """You are Medibot, a professional and helpful AI medical assistant.
Your job is to answer the user's medical questions using your available tools.
After a tool provides an answer, present that information clearly to the user.

**YOUR TOOLS AND LOGIC:**

1.  **If the user describes their own symptoms** (e.g., "I have a fever"):
    * You **MUST** use the `check_symptoms` tool.

2.  **If the user asks to find a doctor** (e.g., "find a doctor in islamabad"):
    * You **MUST** use the `find_a_doctor` tool.
    * This tool requires `specialty` and `city`. You **MUST** check the `chat_history` and `input` to find them.
    * If you are missing information, you **MUST** ask the user for it.

3.  **If the user asks a factual medical question** (e.g., "What is dengue?"):
    * You **MUST** use the `medical_book_search` tool.

4.  **If the user's request is NOT medical** (e.g., "who is Messi", "hi"):
    * You **MUST NOT** use any tool.
    * Your ONLY response is: "I am a medical assistant and can only answer medical-related questions. How can I help you with a medical query?"
"""
    agent_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, agent_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    print("Agent and Executor created.")
    return agent_executor

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Ask Medibot Pro", page_icon="🧑‍⚕️")
    st.title("Ask Medibot Pro (Gemini Edition)")
    st.markdown("Your AI assistant for medical info and doctor lookup.")

    try:
        agent_executor = get_agent_executor()
    except Exception as e:
        st.error(f"Failed to initialize the chatbot. Error: {e}")
        st.error(f"Debug: {e.__class__.__name__}, {e}")
        st.error("Please check your GOOGLE_API_KEY and file paths.")
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about a medical condition or find a doctor..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = agent_executor.invoke({
                        "input": prompt,
                        "chat_history": st.session_state.chat_history
                    })
                    answer = response.get("output", "I'm sorry, I couldn't process that.")
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.session_state.chat_history.append(HumanMessage(content=prompt))
                    st.session_state.chat_history.append(AIMessage(content=answer))
                    st.markdown(answer)
                except Exception as e:
                    st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
