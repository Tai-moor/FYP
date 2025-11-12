import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv, find_dotenv

# Load the .env file
load_dotenv(find_dotenv())

# --- Configuration ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL_NAME = "llama-3.1-8b-instant" # A fast and capable model
DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer.
Dont provide anything out of the given context

Context: {context}
Question: {input}

Start the answer directly. No small talk please.
"""
# ---------------------

def main():
    # 1. Initialize the LLM
    llm = ChatGroq(model_name=GROQ_MODEL_NAME, api_key=GROQ_API_KEY)

    # 2. Set up the Custom Prompt
    prompt = ChatPromptTemplate.from_template(CUSTOM_PROMPT_TEMPLATE)

    # 3. Load the Vector Database
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    print("Loading vector database...")
    db = FAISS.load_local(
        DB_FAISS_PATH, 
        embedding_model, 
        allow_dangerous_deserialization=True # This is needed to load the FAISS index
    )
    
    # 4. Create the Retriever
    retriever = db.as_retriever(search_kwargs={'k': 3}) # Get top 3 results

    # 5. Create the Document Chain (handles context and prompt)
    document_chain = create_stuff_documents_chain(llm, prompt)

    # 6. Create the Retrieval Chain (handles user query and retrieval)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    print("Chatbot is ready. Type your query.")
    
    # 7. Start a loop to ask questions
    while True:
        user_query = input("\nWrite Query Here (or 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        if not user_query:
            continue

        # 8. Invoke the chain
        response = retrieval_chain.invoke({'input': user_query})

        # 9. Print the response
        print("\n--- RESULT ---")
        print(response["answer"])
        print("\n--- SOURCES ---")
        
        # Format and print the sources
        sources = response.get("context", [])
        if sources:
            pages = set()
            for doc in sources:
                # Extract page number from metadata
                page = doc.metadata.get("page", "Unknown")
                pages.add(str(page))
            print(f"Information gathered from pages: {', '.join(sorted(pages))}")
        else:
            print("No sources found.")
        print("---------------")

if __name__ == "__main__":
    main()