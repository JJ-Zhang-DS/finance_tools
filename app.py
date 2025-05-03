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
import pandas as pd
import matplotlib.pyplot as plt

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

    st.title("S&P 500 Investment Simulator")

    # User input
    start_year = st.number_input(
        "Enter the starting year (between 1924 and 2050):",
        min_value=1924, max_value=2050, value=2020, step=1
    )

    # Load data
    CSV_PATH = "data/spy500_history.csv"
    df = pd.read_csv(CSV_PATH)
    df = df[["Year", "Annual_Return", "Dividend_Ratio"]]
    df = df.dropna(subset=["Year"]).copy()
    df["Year"] = df["Year"].astype(int)

    # Prepare simulation years
    years = list(range(start_year, 2051))

    # Build a lookup for returns
    returns = {}
    for _, row in df.iterrows():
        year = int(row["Year"])
        returns[year] = {
            "annual_return": float(row.get("Annual_Return", 0)) / 100,
            "dividend_ratio": float(row.get("Dividend_Ratio", 0)) / 100,
        }

    # Simulation
    INITIAL_DEPOSIT = 100_000
    DEFAULT_ANNUAL_RETURN = 0.10
    DEFAULT_DIVIDEND_RATIO = 0.02
    LAST_DATA_YEAR = 2025
    END_YEAR = 2050
    balance = INITIAL_DEPOSIT
    results = []
    for year in years:
        if year in returns and year <= LAST_DATA_YEAR:
            annual_return = returns[year]["annual_return"]
            dividend_ratio = returns[year]["dividend_ratio"]
        else:
            annual_return = DEFAULT_ANNUAL_RETURN
            dividend_ratio = DEFAULT_DIVIDEND_RATIO
        total_return = annual_return + dividend_ratio
        balance = balance * (1 + total_return)
        results.append({
            "Year": year,
            "Annual Return": f"{annual_return*100:.2f}%",
            "Dividend Ratio": f"{dividend_ratio*100:.2f}%",
            "Total Return": f"{total_return*100:.2f}%",
            "Balance": balance
        })

    results_df = pd.DataFrame(results)

    # Show table
    st.subheader("Simulation Results")
    st.dataframe(results_df.style.format({"Balance": "${:,.2f}"}), use_container_width=True)

    # Plot
    st.subheader("Balance Over Time")
    fig, ax = plt.subplots()
    ax.plot(results_df["Year"], results_df["Balance"], marker="o")
    ax.set_xlabel("Year")
    ax.set_ylabel("Balance ($)")
    ax.set_title("S&P 500 Investment Growth")
    ax.grid(True)
    st.pyplot(fig)

if __name__ == "__main__":
    main()