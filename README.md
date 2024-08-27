

---


### Overview

The **Tata Elxsi AI Document Assistant** is a Streamlit-based web application designed to assist users in retrieving and analyzing information from documents. It supports various file types, including PDFs, PowerPoint presentations, Excel files, and Word documents, making it ideal for organizations with large document repositories. The app uses state-of-the-art AI techniques, including LangChain for text chunking, HuggingFace for embeddings, FAISS for vector storage, and Google Gemini (Google Generative AI) for answering user queries.

### Features

- **Document Indexing:** Automatically extracts text from documents (PDF, PPTX, XLSX, DOCX) and indexes them for fast retrieval.
- **AI-powered Search:** Uses embeddings and a vector store to retrieve the most relevant documents based on user queries.
- **Contextual Answers:** The app answers user queries using the Google Gemini model and handles out-of-context queries gracefully.
- **Download Functionality:** Users can download the original documents directly from the interface.
- **User-friendly Interface:** The app offers a clean, intuitive UI with custom CSS styling.

### Tech Stack

- **Frontend:** Streamlit
- **Backend:** Python
- **Libraries:** 
  - `langchain_community` for text chunking and embedding.
  - `FAISS` for vector storage and search.
  - `PyPDF2`, `pptx`, `openpyxl`, `python-docx` for text extraction.
  - `Google Generative AI` for natural language processing and answering queries.

### Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/tata-elxsi-ai-document-assistant.git
   cd tata-elxsi-ai-document-assistant
   ```

2. **Install Dependencies:**
   Ensure you have Python 3.8 or higher installed. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. **API Key Setup:**
   Create a `.env` file in the root directory and add your Google Generative AI API key:
   ```bash
   GOOGLE_GEMINI_API_KEY=your_api_key_here
   ```

4. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

### Configuration

- **Document Directory:** Update the `root_dir` variable in the `process_query()` function to point to the directory where your documents are stored.
  ```python
  root_dir = "path_to_your_document_directory"
  ```

### Usage

1. **Upload Documents:** Ensure your document directory is populated with the supported file types (PDF, PPTX, XLSX, DOCX).
2. **Ask Questions:** Use the input box to type your queries. The AI will retrieve relevant documents and answer your questions based on the content of those documents.
3. **Download Documents:** Click the download button next to any relevant document to download the original file.

### Project Structure

- **app.py:** Main application file containing the Streamlit code and business logic.
- **requirements.txt:** Lists all Python libraries required to run the application.
- **.env:** File to store sensitive environment variables, such as API keys (not included in the repository for security reasons).

### Customization

- **Model Configuration:** The `initialize_model()` function in `app.py` is where the Google Gemini model is configured. You can adjust model parameters if needed.
- **Chunking Strategy:** The `chunk_text()` function allows you to modify how documents are split into chunks for processing. You can adjust the `chunk_size` and `chunk_overlap` parameters to suit your needs.

### Contributing

If you'd like to contribute to this project, please fork the repository and use a feature branch. Pull requests are warmly welcome.

### License

This project is licensed under the MIT License.

---
