import os
import streamlit as st
import csv
import re
from dotenv import load_dotenv, find_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.tools import Tool, tool
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# =============================
# Environment & Config
# =============================
load_dotenv(find_dotenv())

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
AGENT_MODEL = "gemini-2.0-flash"
VECTORSTORE_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DOCTORS_CSV = "data/doctors.csv"

SAFETY_SETTINGS = {0: 0, 1: 0, 2: 0, 3: 0}

# =============================
# Helper Functions
# =============================
def is_greeting(text: str) -> bool:
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    return any(greet in text.lower() for greet in greetings)

def is_irrelevant(text: str) -> bool:
    """
    Returns True if the message is clearly off-topic (non-medical),
    otherwise False. Disease names are allowed.
    """
    text_lower = text.lower()
    medical_keywords = [
        "doctor", "symptom", "fever", "disease", "medical", "pain",
        "illness", "headache", "cancer", "diabetes", "covid", "flu"
    ]
    return not any(keyword in text_lower for keyword in medical_keywords)

def detect_disease_query(text: str):
    """
    Detects if user mentions a disease-related query.
    Returns:
      - str: disease name if specified
      - None: if general disease question without name
      - False: if not a disease query
    """
    text_lower = text.lower()
    general_patterns = [
        r"i have an issue",
        r"i have a question about (a )?disease",
        r"i want to know about (a )?disease",
        r"tell me about a disease",
    ]
    for pattern in general_patterns:
        if re.search(pattern, text_lower):
            return None

    match = re.search(r"(?:about|on|regarding|for)\s+([a-zA-Z\s]+)", text_lower)
    if match:
        disease_name = match.group(1).strip()
        if len(disease_name.split()) <= 4:
            return disease_name

    return False

# =============================
# Tools
# =============================
@tool(description="Find a doctor by specialty and city.")
def doctor_lookup(specialty: str, city: str) -> str:
    if not specialty or not city:
        return "Please provide both the specialty and city."
    results = []
    try:
        with open(DOCTORS_CSV, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if specialty.lower() in row['specialty'].lower() and city.lower() in row['city'].lower():
                    results.append(
                        f"Name: {row['name']}, Specialty: {row['specialty']}, "
                        f"City: {row['city']}, Address: {row['address']}, Phone: {row['phone']}"
                    )
        return "\n".join(results) if results else f"No doctors found for '{specialty}' in '{city}'."
    except Exception as e:
        return f"Error reading doctor data: {e}"

@tool(description="Check symptoms and suggest possible conditions.")
def symptom_checker(symptoms: str) -> str:
    llm = ChatGoogleGenerativeAI(
        model=AGENT_MODEL,
        google_api_key=GOOGLE_API_KEY,
        safety_settings=SAFETY_SETTINGS
    )
    prompt_template = """
Provide **up to 3 possible medical conditions** for the user's symptoms.
Include a confidence rating: High, Medium, or Low.

Format:

- Condition: [Name], Confidence: [High/Medium/Low]

Symptoms: "{user_symptoms}"
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm
    response = chain.invoke({"user_symptoms": symptoms})
    return response.content

# =============================
# Vector Store
# =============================
@st.cache_resource
def load_vector_store():
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(VECTORSTORE_PATH, embedding, allow_dangerous_deserialization=True)

# =============================
# Agent Executor
# =============================
@st.cache_resource
def initialize_agent():
    llm = ChatGoogleGenerativeAI(model=AGENT_MODEL, google_api_key=GOOGLE_API_KEY, safety_settings=SAFETY_SETTINGS)
    retriever = load_vector_store().as_retriever(search_kwargs={"k": 3})
    document_chain = create_stuff_documents_chain(llm, ChatPromptTemplate.from_template(
        "Answer the question based on the context below:\nContext: {context}\nQuestion: {input}\nAnswer:"
    ))
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    def medical_search(query: str):
        return retrieval_chain.invoke({"input": query})

    tools = [
        Tool(name="medical_book_search", func=medical_search, description="Retrieve factual medical info."),
        doctor_lookup,
        symptom_checker
    ]

    system_prompt = """
You are Medibot, a professional medical assistant. 
Handle medical queries using your tools. 
Respond politely to greetings and irrelevant queries.
"""
    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, tools, agent_prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# =============================
# Streamlit App
# =============================
def main():
    st.set_page_config(page_title="Medibot Pro", page_icon="🧑‍⚕️")
    st.title("Medibot Pro")
    st.markdown("AI assistant for medical questions, symptoms, and doctor lookup.")

    agent_executor = initialize_agent()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pending_disease" not in st.session_state:
        st.session_state.pending_disease = None

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("Enter your query..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                try:
                    # Greeting only if no pending disease
                    if is_greeting(user_input) and not st.session_state.pending_disease:
                        reply = "Hello! How can I help you with a medical query today?"

                    # Pending disease clarification
                    elif st.session_state.pending_disease:
                        disease_name = st.session_state.pending_disease
                        st.session_state.pending_disease = None
                        query = f"What can you tell me about {disease_name}? {user_input}"
                        response = agent_executor.invoke({
                            "input": query,
                            "chat_history": st.session_state.chat_history
                        })
                        reply = response.get("output", f"No information found for {disease_name}.")

                    # Detect disease query
                    else:
                        disease_check = detect_disease_query(user_input)
                        if isinstance(disease_check, str):
                            st.session_state.pending_disease = disease_check
                            reply = f"What specifically would you like to know about {disease_check}?"
                        elif disease_check is None:
                            reply = "Could you specify which disease you are asking about?"
                        # Only irrelevant if no disease detected
                        elif is_irrelevant(user_input):
                            reply = "I can only answer medical-related questions. Could you please ask a medical question?"
                        else:
                            response = agent_executor.invoke({
                                "input": user_input,
                                "chat_history": st.session_state.chat_history
                            })
                            reply = response.get("output", "I could not process your request.")

                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.session_state.chat_history.append(HumanMessage(content=user_input))
                    st.session_state.chat_history.append(AIMessage(content=reply))
                    st.markdown(reply)
                except Exception as e:
                    st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
