import os
import boto3
import pdfplumber 
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

import uuid
# Load environment variables
load_dotenv()

# Initialize S3 client
s3_client = boto3.client('s3')

AWS_BUCKET_NAME = os.getenv('AWS_BUCKET_NAME')
AWS_S3_KEY = os.getenv('AWS_S3_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

chat_model = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4")


def download_pdf_from_s3():
    """Download the PDF file from S3 and save it locally."""
    try:
        response = s3_client.get_object(Bucket=AWS_BUCKET_NAME, Key=AWS_S3_KEY)
        pdf_data = response['Body'].read()

        # Extract filename from S3 key
        file_name = os.path.basename(AWS_S3_KEY)
        file_path = f"./tmp/{file_name}"

        # Save the file locally
        with open(file_path, 'wb') as f:
            f.write(pdf_data)

        print(f"PDF downloaded and saved as: {file_path}")
        return file_path

    except Exception as e:
        print(f"Error downloading file: {e}")
        return None

def extract_text_with_chunks(pdf_path):
    try :
        loader = PDFPlumberLoader(pdf_path)
        total_texts = loader.load()
        chunked_data = []
        for doc in total_texts:  # Each `doc` is a `Document` object
            # for each page in the document chunk
            chunked_text = split_text(doc.page_content)
            # generate the embeddings
            chunked_data.append(chunked_text)
        return chunked_data
        
    except Exception as e:
        print(f"Error extracting text: {e}")
        return None

def split_text(text):
    try :
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Split the text into chunks of 5000 characters
            chunk_overlap=100,  # Overlap of 100 characters between chunks
        )
        print(text_splitter)
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        print(f"Error splitting text: {e}")
        return None
    
def check_or_create_pinecone_index() :
    try :
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index_name='rag-learnings'
        for index in pc.list_indexes():
            if index.name == index_name:
                return pc.Index(index_name)
        
        spec = ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
        index_created = pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=spec
        )
        return pc.Index(index_name)
    except Exception as e:
        print(f"Error creating Pinecone index: {e}")
        return None
    

def create_embedding(text):
    """Generate embedding using OpenAI."""
    response = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        model="text-embedding-ada-002"  # OpenAI’s embedding model
    )
    response = response.embed_query(text)
    return response
    
def create_vectors(chunked_data, index) :
    vectors = []
    document_id = str(uuid.uuid4())
    try :
        for chunk in chunked_data:
            chunk_text = " ".join(chunk) if isinstance(chunk, list) else chunk
            embeddings = create_embedding(chunk[0])
            vector_object = {
                "id" : str(uuid.uuid4()),
                "values" : embeddings,
                "metadata" : {
                    "user_id" : "",
                    "document_id" : document_id,
                    "text" : chunk_text
                }
            }
            vectors.append(vector_object)
        return vectors
    except Exception as e:
        print(f"Error creating vectors: {e}")
        return None
        
def main() :
    try :
        pdf_path = download_pdf_from_s3()
        total_chunks = extract_text_with_chunks(pdf_path)
        print(f"Total chunks extracted: {len(total_chunks)}")

        pinecone_index = check_or_create_pinecone_index()
        
        if len(total_chunks) == 0:
            print("No text extracted from the PDF.")
            return None
        generated_vectors = create_vectors(total_chunks, pinecone_index)

        if len(generated_vectors) == 0:
            print("No vectors generated.")
            return None
        
        print("Vectors generated successfully.")
        # upsert the vectors
        pinecone_index.upsert(vectors=generated_vectors)
        
    except Exception as e:
        print(f"Error in main: {e}")
        return None

def format_context(search_results):
    """Format search results into a context string for the LLM."""
    if 'matches' not in search_results:
        return "No relevant information found."
    
    context_parts = []
    for match in search_results['matches']:
        if match.score >= 0.7:  # Only include relevant matches
            text = match.metadata.get('text', '')
            context_parts.append(text)
    
    return "\n\n".join(context_parts)

def query_handler() :
    try :
        user_query = input("Enter the query: ")
        embedding_model = OpenAIEmbeddings(
            api_key=OPENAI_API_KEY,
            model="text-embedding-ada-002"
        )
        query_embedding = embedding_model.embed_query(user_query)
        pinecone_index = check_or_create_pinecone_index()
        query_params = {
        "vector": query_embedding, 
        "top_k": 5, 
        "include_metadata": True
        }

        search_results = pinecone_index.query(**query_params)
        context = format_context(search_results)
        if context is None:
            print("No context found.")
            return None
        answer = generate_answer(context, user_query)
        print(f"Answer: {answer}")


    except Exception as e:
        print(f"Error in query_handler: {e}")
        return None
    
def generate_answer(context, query)  :
    try :
        prompt_template = PromptTemplate(
        template="You are an AI assistant. Use the following context to answer the query.\n\nContext:\n{context}\n\nUser Query: {query}\nAnswer:",
        input_variables=["context", "query"]
        )

        final_prompt = prompt_template.format(context=context, query=query)
        response = chat_model.invoke(final_prompt)

        return response
    except Exception as e:
        print(f"Error in generate_answer: {e}")
        return None

if __name__ == "__main__":
    query_handler()
