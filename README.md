MediBot Pro: AI-Powered Healthcare Agent 🩺
MediBot Pro is an advanced, hybrid AI agent designed to assist users with medical inquiries. Unlike simple chatbots, MediBot utilizes an Agentic Architecture powered by Google Gemini, capable of intelligent decision-making to route user queries to specific tools.

It combines Retrieval-Augmented Generation (RAG) for factual medical knowledge, Deterministic Search for finding doctors, and LLM Analysis for symptom checking.

🚀 Key Features
🤖 Intelligent Agent Brain: Uses Google Gemini to analyze user intent and strictly enforce guardrails (rejecting non-medical queries).

📚 Medical Knowledge Base (RAG): Retrieves accurate, factual answers from a vector database created from trusted medical encyclopedias.

🏥 Doctor Finder (Hybrid Tool): A deterministic tool that searches a local database (doctors.csv) to find specialists in specific cities (e.g., "Cardiologist in Islamabad").

ea Symptom Checker: Analyzes user-described symptoms and suggests possible conditions with mandatory safety disclaimers.

🧠 Context Awareness: Maintains chat history to understand follow-up questions (e.g., remembering the city when the user provides the specialty later).

🛠️ Tech Stack
Frontend: Streamlit (Python)

LLM Orchestration: LangChain

AI Model: Google Gemini (via langchain-google-genai)

Vector Database: FAISS (Facebook AI Similarity Search)

Embeddings: HuggingFace (sentence-transformers/all-MiniLM-L6-v2)

Data Handling: Python csv and pypdf modules

📂 Project Structure
medical-chatbot/
├── data/
│   ├── doctors.csv          # Local database of doctors
│   └── medical_book.pdf     # Source PDF for the RAG system
├── vectorstore/
│   └── db_faiss/            # The generated vector database (AI Memory)
├── .env                     # API Keys (Not committed to Git)
├── medibot.py               # Main Application (Streamlit + Agent Logic)
├── create_memory_for_LLM.py # Script to ingest PDF and build Vector DB
└── requirements.txt         # Project dependencies

⚙️ Setup & Installation
1. Clone the Repository

git clone <your-repo-link>
cd medical-chatbot
2. Create a Virtual Environment
It is recommended to use a clean environment to avoid conflicts.

# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
Bash

pip install streamlit python-dotenv langchain langchain-google-genai langchain-community langchain-huggingface faiss-cpu sentence-transformers pypdf
4. Configure API Keys
Create a .env file in the root directory and add your Google API key:

Code snippet

GOOGLE_API_KEY=AIzaSy...YourKeyHere
🚀 Usage
Step 1: Build the Memory (Run once)
If you haven't created the vector database yet, run this script to process your PDF:

python create_memory_for_LLM.py
Step 2: Run the Chatbot
Launch the Streamlit application:

streamlit run medibot.py
🧪 Example Queries
1. Doctor Finding (Local Data Tool):

"Find a cardiologist in Islamabad" "I need a doctor." -> (Bot asks for specialty) -> "Dermatologist" -> (Bot asks for city) -> "Lahore"

2. Medical Knowledge (RAG Tool):

"What are the symptoms of Dengue?" "How do you treat a migraine?"

3. Symptom Analysis (AI Tool):

"I have a fever and a bad headache."

4. Guardrails (Safety Check):

"Who is Lionel Messi?" (Bot refuses to answer non-medical questions)

⚠️ Disclaimer
MediBot Pro is an AI prototype for educational purposes only. It is not a licensed medical professional. All advice provided by the bot should be verified by a certified doctor. In case of a medical emergency, contact local emergency services immediately.
