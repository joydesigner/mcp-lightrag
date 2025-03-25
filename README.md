# Blog on this project
https://daybreak.hashnode.dev/mcp-server-setup-part-1-building-a-local-rag-with-deepseek-for-enhanced-security-and-compliance
# Multi-Contextual Processing (MCP) Server

This repository contains a Multi-Contextual Processing (MCP) server implementation that uses the DeepSeek-R1 model for processing and querying documents using a RAG framework - lightRAG, it can manipulate both text and PDF files. The server supports multiple query modes and provides a RESTful API interface.

## Features

- **Multiple Query Modes**:
  - Naive Search: Direct document search
  - Local Search: Context-aware local search
  - Global Search: Comprehensive document search
  - Hybrid Search: Combines local and global search strategies

- **PDF Support**: Built-in support for processing PDF documents using pdfplumber
- **FastAPI Integration**: Modern, fast web framework for building APIs
- **Async Processing**: Asynchronous document processing and querying
- **Error Handling**: Robust error handling and retry mechanisms
- **Health Monitoring**: Built-in health check endpoint

## Prerequisites

- Python 3.12 or higher
- Virtual environment (recommended)
- DeepSeek API key
- Ollama running locally (for embeddings)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

1. Set up environment variables:
```bash
export DEEPSEEK_API_KEY=your_api_key_here
```

2. Ensure Ollama is running locally (for embeddings):
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags
```

## Usage

1. Start the server:
```bash
python mcp_server.py
```

2. The server will be available at `http://localhost:8000`

### API Endpoints

- `POST /insert`: Insert a new document
  ```bash
  curl -X POST "http://localhost:8000/insert" \
       -H "Content-Type: application/json" \
       -d '{"content": "Your document content here"}'
  ```

- `POST /query`: Query the documents
  ```bash
  curl -X POST "http://localhost:8000/query" \
       -H "Content-Type: application/json" \
       -d '{"query": "Your query here", "mode": "naive"}'
  ```

- `GET /modes`: List available query modes
  ```bash
  curl "http://localhost:8000/modes"
  ```

- `GET /health`: Check server health
  ```bash
  curl "http://localhost:8000/health"
  ```

### Processing PDF Files

You can process PDF files using the provided script:

```python
import pdfplumber

pdf_path = "./your_document.pdf"
pdf_text = ""
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        pdf_text += page.extract_text() + "\n"
# The text can then be processed by the RAG system
```

## Dependencies

- fastapi>=0.104.1: Web framework
- uvicorn>=0.24.0: ASGI server
- pydantic>=2.4.2: Data validation
- tenacity>=8.2.3: Retry mechanism
- python-dotenv>=1.0.0: Environment variable management
- httpx>=0.25.1: HTTP client
- numpy<2.0.0: Numerical computations
- scipy>=1.8.0,<1.14.0: Scientific computations
- tqdm>=4.66.1: Progress bars
- pdfplumber>=0.10.3: PDF processing

## API Documentation

Once the server is running, you can access the interactive API documentation:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Error Handling

The server includes comprehensive error handling:
- API connection issues
- PDF processing errors
- Query execution errors
- Invalid mode selections

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

[Add your license information here]

## Contact

[Add your contact information here] 