import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.ollama import ollama_embed
from lightrag.utils import EmbeddingFunc
from tenacity import retry, stop_after_attempt, wait_exponential
import pdfplumber


# Define the working directory for lightRAG
WORKING_DIR = "./mybook"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
    
# Define embedding dimensions
EMBEDDING_DIM = 768  # Updated to match nomic-embed-text's actual output dimension

# integrate with DeepSeek API
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs) -> str:
    try:
        response = await openai_complete_if_cache(
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            prompt=prompt,
            system_prompt=system_prompt or "You are an prompt engineering expert, you can help me to generate a prompt for a given task.",
            history_messages=history_messages or [],
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.siliconflow.cn/v1",
            max_tokens=1024,  # Increased max tokens
            temperature=0.3,  # Lower temperature for more consistent responses
            top_p=0.9,  # Added top_p parameter
            presence_penalty=0.1,  # Added presence penalty
            frequency_penalty=0.1,  # Added frequency penalty
        )
        
        if not response or response.strip() == "":
            raise ValueError("Empty response received from API")
            
        return response
    except Exception as e:
        print(f"Error in LLM model function: {str(e)}")
        raise

async def test_deepseek_api():
    """Test the DeepSeek API connection and response"""
    print("\nTesting DeepSeek API connection...")
    try:
        # Test with a simple prompt
        test_prompt = "Say hello and confirm you are working."
        print(f"Sending test prompt: {test_prompt}")
        
        response = await llm_model_func(test_prompt)
        print("\nAPI Response:")
        print(response)
        print("\nAPI test successful!")
        return True
    except Exception as e:
        print(f"\nAPI test failed with error: {str(e)}")
        return False

async def custom_embed(texts):
    try:
        embeddings = await ollama_embed(
            texts,
            embed_model="nomic-embed-text",
            host="http://localhost:11434"  # Using localhost as default
        )
        # Verify embedding dimensions
        if embeddings and len(embeddings) > 0 and len(embeddings[0]) != EMBEDDING_DIM:
            raise ValueError(f"Unexpected embedding dimension: {len(embeddings[0])}, expected {EMBEDDING_DIM}")
        return embeddings
    except Exception as e:
        print(f"Error in embedding function: {str(e)}")
        raise

# Initialize the lightRAG instance
async def initialize_rag():
    try:
        # Clear existing storage to avoid dimension mismatch with previous data
        if os.path.exists(WORKING_DIR):
            import shutil
            shutil.rmtree(WORKING_DIR)
            os.mkdir(WORKING_DIR)
            
        embedding_func = EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=512,
            func=custom_embed,
        )
        
        rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=llm_model_func,
            embedding_func=embedding_func,
        )

        await rag.initialize_storages()
        await initialize_pipeline_status()

        return rag
    except Exception as e:
        print(f"Error initializing RAG: {str(e)}")
        raise

async def process_book(rag, content):
    """Process the book content asynchronously"""
    try:
        print("Starting book processing...")
        await rag.ainsert(content)
        print("Book processing completed successfully")
    except Exception as e:
        print(f"Error processing book: {str(e)}")
        raise

async def query_rag(rag, query, mode):
    """Execute a query asynchronously"""
    try:
        print(f"Processing query in {mode} mode...")
        response = await rag.aquery(query, param=QueryParam(mode=mode))
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error processing query: {str(e)}")

async def main_async():
    try:
        # First test the DeepSeek API
        api_test_success = await test_deepseek_api()
        if not api_test_success:
            print("API test failed. Please check your API key and connection.")
            return

        # Initialize RAG instance
        rag = await initialize_rag()

        # Process PDF file if it exists
        pdf_path = "./State_of_EV_2024.pdf"  # or "Constitution_of_India.pdf"
        if os.path.exists(pdf_path):
            print(f"\nProcessing PDF file: {pdf_path}")
            pdf_text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    pdf_text += page.extract_text() + "\n"
            print(f"Extracted {len(pdf_text)} characters from PDF")
            await process_book(rag, pdf_text)
        # Fallback to book.txt if PDF doesn't exist
        elif os.path.exists("./book.txt"):
            print("\nProcessing book.txt...")
            with open("./book.txt", "r", encoding="utf-8") as f:
                content = f.read()
                print(f"Read {len(content)} characters from book.txt")
                await process_book(rag, content)
        else:
            print("Error: Neither paper.pdf nor book.txt found in the current directory")
            return

        # Example queries
        queries = [
            ("What are the top themes in this story?", "naive"),
            ("What are the top themes in this story?", "local"),
            ("What are the top themes in this story?", "global"),
            ("What are the top themes in this story?", "hybrid")
        ]

        for query, mode in queries:
            print(f"\nQuery: {query}")
            print(f"Mode: {mode}")
            await query_rag(rag, query, mode)

    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()