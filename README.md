# Finance Analysis Project

A comprehensive financial analysis toolkit that includes investment simulations, retirement planning, and PDF document analysis capabilities.

## Features

### 1. S&P 500 Investment Simulator
- Historical S&P 500 performance simulation
- Customizable investment periods
- Dividend reinvestment analysis
- Visual representation of investment growth
- Interactive web interface using Streamlit

### 2. investment analysis
- Upload and analyze PDF documents
- Extract and display tables from PDFs
- AI-powered document Q&A using LangChain and OpenAI
- Interactive chat interface for document queries

### 3. Retirement Planning Tools
- Annuity vs. S&P 500 comparison
- Historical market simulation
- Retirement income projections
- Financial calculator utilities

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/finance-analysis.git
cd finance-analysis
```

2. Create and activate a virtual environment:
```bash
python -m venv fin_env
source fin_env/bin/activate  # On Windows: fin_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the root directory with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Running the Web Application
```bash
streamlit run app.py
```

### Running Simulations
- S&P 500 Simulation: `python spy_simulation.py`
- Retirement Simulation: `python test_retirement_simulation.py`

## Project Structure

- `app.py`: Main Streamlit web application
- `spy_simulation.py`: S&P 500 investment simulation
- `test_retirement_simulation.py`: Retirement planning simulation
- `finance_calculator.py`: Financial calculation utilities
- `data/`: Directory containing historical market data
- `*.ipynb`: Jupyter notebooks for analysis and visualization

## Requirements

- Python 3.8+
- Streamlit
- LangChain
- OpenAI API
- Pandas
- Matplotlib
- PDF processing libraries (pdfplumber, PyPDF2)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 