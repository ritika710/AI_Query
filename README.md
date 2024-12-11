# LLM-Based PDF Chatbot Project

This project is a Streamlit-based application that uses PDFs to extract information and employs the Gemini API to answer user queries. The application leverages LangChain, FAISS, and Google Generative AI to create a conversational system.

## Features
- Extract text from uploaded PDF documents.
- Generate embeddings using Google Generative AI.
- Build and save a FAISS vector store for efficient similarity searches.
- Use a Gemini LLM (Large Language Model) to answer questions based on the extracted content.
- Send responses via email if they exceed a specific length.

## Requirements

### Libraries
Install the required Python libraries:

pip install streamlit PyPDF2 langchain langchain-google-genai google-generativeai python-dotenv faiss-cpu

## Steps to Run the Application

1. **Clone the Repository:**
   - Clone the repository and navigate to the project folder.

2. **Create a .env File:**
   - Add your Google API key to a `.env` file in the project directory:
     - `GOOGLE_API_KEY=<your-google-api-key>`

3. **Install Dependencies:**
   - Install all necessary libraries by running the command:
     - `pip install -r requirements.txt`

4. **Run the Application:**
   - Start the Streamlit application:
     - `streamlit run app.py`

5. **Upload PDF Files:**
   - Use the file uploader in the Streamlit interface to upload PDFs.
   - Click "Submit & Process" to extract text and create the vector store.

6. **Ask Questions:**
   - Enter a query in the input box to get an answer based on the uploaded PDF content.

## Code Functionality

1. **Extracting PDF Content:**
   - Reads and extracts text from uploaded PDFs using PyPDF2.

2. **Text Splitting and Embeddings:**
   - Splits the extracted text into chunks using `RecursiveCharacterTextSplitter`.
   - Generates embeddings using `GoogleGenerativeAIEmbeddings`.

3. **Vector Store:**
   - Builds and saves a FAISS vector store for efficient similarity searches.

4. **Question Answering:**
   - Uses a Gemini API-based conversational model to generate answers.

5. **Email Responses:**
   - Sends long responses via email using `smtplib`.

