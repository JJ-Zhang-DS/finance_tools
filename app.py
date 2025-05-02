import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import tempfile
from typing import Dict, List
import pdfplumber

# Load environment variables
load_dotenv()

class PDFAnalyzer:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )
        self.llm = OpenAI(temperature=0.3)

    def extract_tables(self, pdf_path: str) -> List[Dict]:
        """Extract tables using pdfplumber"""
        tables = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_tables = page.extract_tables()
                if page_tables:
                    tables.append({
                        'page': page_num,
                        'tables': page_tables
                    })
        return tables

    def process_document(self, pdf_path: str):
        """Process PDF document"""
        # Extract tables
        tables = self.extract_tables(pdf_path)

        # Load and split text
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)

        # Create vector store
        vectorstore = FAISS.from_documents(chunks, self.embeddings)

        # Create QA chain
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )

        return {
            'tables': tables,
            'vectorstore': vectorstore,
            'qa_chain': qa_chain
        }

def main():
    st.title("📄 PDF Analysis Assistant")

    # Initialize analyzer
    analyzer = PDFAnalyzer()

    # File upload
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf:
            tmp_pdf.write(uploaded_file.getvalue())
            pdf_path = tmp_pdf.name

            try:
                # Process document
                with st.spinner('Processing document...'):
                    results = analyzer.process_document(pdf_path)
                    st.session_state['results'] = results
                    st.success("Document processed!")

                # Display tables
                if results['tables']:
                    with st.expander("📊 Tables Found"):
                        for page_tables in results['tables']:
                            st.write(f"Page {page_tables['page']}:")
                            for i, table in enumerate(page_tables['tables'], 1):
                                st.write(f"Table {i}:")
                                st.dataframe(table)

                # Chat interface
                st.subheader("💬 Ask Questions")
                
                if 'messages' not in st.session_state:
                    st.session_state.messages = []

                # Display chat history
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                # Question input
                if prompt := st.chat_input("Ask about the document"):
                    st.session_state.messages.append({"role": "user", "content": prompt})

                    with st.chat_message("assistant"):
                        response = results['qa_chain']({"question": prompt})
                        st.markdown(response['answer'])
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response['answer']}
                        )

            finally:
                os.unlink(pdf_path)

if __name__ == "__main__":
    main()