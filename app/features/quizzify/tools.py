from typing import List, Tuple, Union
from io import BytesIO
from fastapi import UploadFile
from pypdf import PdfReader
from urllib.parse import urlparse
import requests
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'services')))
from langchain_community.document_loaders import TextLoader, YoutubeLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_vertexai import VertexAIEmbeddings, VertexAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from services.logger import setup_logger
from services.tool_registry import ToolFile
from api.error_utilities import LoaderError, VideoTranscriptError

relative_path = "features/quzzify"

logger = setup_logger(__name__)

def read_text_file(file_path):
    # Get the directory containing the script file
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Combine the script directory with the relative file path
    absolute_file_path = os.path.join(script_dir, file_path)
    
    with open(absolute_file_path, 'r') as file:
        return file.read()

class RAGRunnable:
    def __init__(self, func):
        self.func = func
    
    def __or__(self, other):
        def chained_func(*args, **kwargs):
            # Result of previous function is passed as first argument to next function
            return other(self.func(*args, **kwargs))
        return RAGRunnable(chained_func)
    
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

class UploadPDFLoader:
    def __init__(self, files: List[UploadFile]):
        self.files = files

    def load(self) -> List[Document]:
        documents = []

        for upload_file in self.files:
            with upload_file.file as pdf_file:
                pdf_reader = PdfReader(pdf_file)

                for i, page in enumerate(pdf_reader.pages):
                    page_content = page.extract_text()
                    metadata = {"source": upload_file.filename, "page_number": i + 1}

                    doc = Document(page_content=page_content, metadata=metadata)
                    documents.append(doc)

        return documents

class BytesFilePDFLoader:
    def __init__(self, files: List[Tuple[BytesIO, str]]):
        self.files = files
    
    def load(self) -> List[Document]:
        documents = []
        
        for file, file_type in self.files:
            logger.debug(file_type)
            if file_type.lower() == "pdf":
                pdf_reader = PdfReader(file) #! PyPDF2.PdfReader is deprecated

                for i, page in enumerate(pdf_reader.pages):
                    page_content = page.extract_text()
                    metadata = {"source": file_type, "page_number": i + 1}

                    doc = Document(page_content=page_content, metadata=metadata)
                    documents.append(doc)
                    
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
        return documents

class LocalFileLoader:
    def __init__(self, file_paths: list[str], expected_file_type="pdf"):
        self.file_paths = file_paths
        self.expected_file_type = expected_file_type

    def load(self) -> List[Document]:
        documents = []
        
        # Ensure file paths is a list
        self.file_paths = [self.file_paths] if isinstance(self.file_paths, str) else self.file_paths
    
        for file_path in self.file_paths:
            
            file_type = file_path.split(".")[-1]

            if file_type != self.expected_file_type:
                raise ValueError(f"Expected file type: {self.expected_file_type}, but got: {file_type}")

            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)

                for i, page in enumerate(pdf_reader.pages):
                    page_content = page.extract_text()
                    metadata = {"source": file_path, "page_number": i + 1}

                    doc = Document(page_content=page_content, metadata=metadata)
                    documents.append(doc)

        return documents

class TextFileLoader:
    def __init__(self, files:List[Tuple[BytesIO, str]] ):
        self.files = files
    def load(self) -> List[Document]:
        #documents = []
        #text_loader_kwargs = {'autodetect_encoding':True}
        for file, file_type in self.files:
            logger.debug(f"Loading file of type: {file_type}")
            if file_type.lower() == "txt":
                try:
                    #file.seek(0)
                    loader = TextLoader(file)
                    loaded_documents = loader.load()
                except Exception as e:
                    logger.error(f"Error loading text file: {e}")
                    raise e
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        return loaded_documents

class URLLoader:
    def __init__(self, file_loader=None, expected_file_type="txt", verbose=False):
        self.loader = file_loader
        self.expected_file_type = expected_file_type
        self.verbose = verbose

    def load(self, tool_files: List[ToolFile]) -> List[Document]:
        queued_files = []
        documents = []
        any_success = False
        for tool_file in tool_files:
            try:
                url = tool_file.url
                response = requests.get(url)
                parsed_url = urlparse(url)
                path = parsed_url.path
                if response.status_code == 200:
                    # Read file
                    file_content = BytesIO(response.content)
                    # Check file type
                    file_type = path.split(".")[-1]
                    if file_type != self.expected_file_type:
                        raise LoaderError(f"Expected file type: {self.expected_file_type}, but got: {file_type}")
                    # Append to Queue
                    queued_files.append((file_content,file_type))
                    if self.verbose:
                        logger.info(f"Successfully loaded file from {url}")
                    any_success = True  # Mark that at least one file was successfully loaded
                else:
                    logger.error(f"Request failed to load file from {url} and got status code {response.status_code}")
            except Exception as e:
                logger.error(f"Failed to load file from {url}")
                logger.error(e)
                continue
        # Pass Queue to the file loader if there are any successful loads
        if any_success:
            file_loader = self.loader(queued_files)
            documents = file_loader.load()
            if self.verbose:
                logger.info(f"Loaded {len(documents)} documents")
        if not any_success:
            raise LoaderError("Unable to load any files from URLs")
        return documents
    

class YoutubeTranscriptLoader:
    def __init__(self, video_urls: List[ToolFile]):
        self.video_urls = video_urls

    def load(self) -> List[Document]:
        print("file in class:",self.video_urls)
        for file in self.video_urls:
            try:
                loader = YoutubeLoader.from_youtube_url(file, add_video_info=True)
                docs = loader.load()
                #metadata = {
                 #   "author": docs[0].metadata['author'],
                  #  "title": docs[0].metadata.get("title"),
                   # "length": docs[0].metadata.get("length"),
                #}
            except VideoTranscriptError as e:
                logger.error(f"VideoTranscriptError: {str(e)} for URL: {file}")
                raise
            except Exception as e:
                logger.error(f"An error occurred while processing video at {file}: {str(e)}")
                raise VideoTranscriptError(f"Video transcript might be private or unavailable in 'en' or the URL is incorrect.", file) from e
        return docs

class WebPageLoader:
    def __init__(self, web_urls: List[ToolFile]):
        self.web_urls = web_urls

    def load(self) -> List[Document]:
        #documents = []
        for file in self.web_urls:
            try:
                loader = WebBaseLoader(file)
                docs = loader.load()
            except Exception as e:
                logger.error(f"An error occurred while processing web page at {file}: {str(e)}")
                raise LoaderError(f"Error loading web page: {file}") from e
        return docs


class RAGpipeline:
    def __init__(self, 
                 youtube_loader=None,
                 txt_url_loader=None,
                 web_loader=None, 
                 splitter=None, 
                 vectorstore_class=None, 
                 embedding_model=None, 
                 verbose=False):
        default_config = {
            "text_url_loader": URLLoader(file_loader=TextFileLoader,verbose=verbose), # Creates instance on call
            "web_loader":WebPageLoader,
            "youtube_loader": YoutubeTranscriptLoader, # Creates instance on call
            "splitter": RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100),
            "vectorstore_class": Chroma,
            "embedding_model": VertexAIEmbeddings(model='textembedding-gecko', project='multigenquiz')
        }
        self.txt_url_loader = txt_url_loader or default_config["text_url_loader"]
        self.youtube_loader = youtube_loader or default_config["youtube_loader"]
        self.web_loader = web_loader or default_config["web_loader"]
        self.splitter = splitter or default_config["splitter"]
        self.vectorstore_class = vectorstore_class or default_config["vectorstore_class"]
        self.embedding_model = embedding_model or default_config["embedding_model"]
        self.verbose = verbose

    def load_documents(self, files) -> List[Document]:
        documents = []
        #pdf_file_urls = [file for file in files if isinstance(file, ToolFile) and file.url.lower().endswith('.pdf')]
        text_file_urls = [file for file in files if isinstance(file, ToolFile) and file.url.lower().endswith('.txt')]
        #print(type(files[0]))
        youtube_urls = [file.url for file in files if isinstance(file, ToolFile) and 'youtube' in file.url.lower()]
        web_urls = [
            file.url for file in files if isinstance(file, ToolFile) and file.url.lower().startswith('http') 
            and not text_file_urls and not youtube_urls
        ]
        # Check for multiple formats
        
        if text_file_urls:
            print("Text URLs found:", text_file_urls)
            try:
                text_documents = self.txt_url_loader.load(text_file_urls)
                documents.extend(text_documents)
                print("Text Documents Found:", text_documents)
            except Exception as e:
                logger.error(f"An error occurred while loading text files: {str(e)}")
                raise LoaderError("Error loading text files") from e

        if youtube_urls:
            print("YouTube URLs found:", youtube_urls)
            try:
                youtube_load_instance = self.youtube_loader(youtube_urls) 
                youtube_documents = youtube_load_instance.load()
                documents.extend(youtube_documents)
                print("Loaded documents from YouTube:", youtube_documents)
            except VideoTranscriptError as e:
                logger.error(f"VideoTranscriptError: {str(e)}")
                raise LoaderError("Error loading YouTube transcripts") from e
            except Exception as e:
                logger.error(f"An error occurred while loading YouTube transcripts: {str(e)}")
                raise LoaderError("Unexpected error loading YouTube transcripts") from e
      
        if web_urls:
            print("Web URLs found:", web_urls)
            try:
                web_loader_instance = self.web_loader(web_urls)
                web_documents = web_loader_instance.load()
                documents.extend(web_documents)
                print("Loaded documents from web URLs:", web_documents)
            except Exception as e:
                logger.error(f"An error occurred while loading web URLs: {str(e)}")
                raise LoaderError("Error loading web URLs") from e
            
        return documents
    
    def split_loaded_documents(self, loaded_documents: List[Document]) -> List[Document]:
        if self.verbose:
            logger.info(f"Splitting {len(loaded_documents)} documents")
            logger.info(f"Splitter type used: {type(self.splitter)}")
            
        total_chunks = []
        chunks = self.splitter.split_documents(loaded_documents)
        total_chunks.extend(chunks)
        
        if self.verbose: logger.info(f"Split {len(loaded_documents)} documents into {len(total_chunks)} chunks")
        
        return total_chunks
    
    def create_vectorstore(self, documents: List[Document]):
        if self.verbose:
            logger.info(f"Creating vectorstore from {len(documents)} documents")
        
        self.vectorstore = self.vectorstore_class.from_documents(documents, self.embedding_model)

        if self.verbose: logger.info(f"Vectorstore created")
        return self.vectorstore
    
    def compile(self):
        # Compile the pipeline
        self.load_PDFs = RAGRunnable(self.load_PDFs)
        self.split_loaded_documents = RAGRunnable(self.split_loaded_documents)
        self.create_vectorstore = RAGRunnable(self.create_vectorstore)
        if self.verbose: logger.info(f"Completed pipeline compilation")
    
    def __call__(self, documents):
        # Returns a vectorstore ready for usage 
        
        if self.verbose: 
            logger.info(f"Executing pipeline")
            logger.info(f"Start of Pipeline received: {len(documents)} documents of type {type(documents[0])}")
        
        pipeline = self.load_PDFs | self.split_loaded_documents | self.create_vectorstore
        return pipeline(documents)

class QuizBuilder:
    def __init__(self, vectorstore, topic, prompt=None, model=None, parser=None, verbose=False):
        default_config = {
            "model": VertexAI(model="gemini-1.0-pro"),
            "parser": JsonOutputParser(pydantic_object=QuizQuestion),
            "prompt": read_text_file("prompt/quizzify-prompt.txt")
        }
        
        self.prompt = prompt or default_config["prompt"]
        self.model = model or default_config["model"]
        self.parser = parser or default_config["parser"]
        
        self.vectorstore = vectorstore
        self.topic = topic
        self.verbose = verbose
        
        if vectorstore is None: raise ValueError("Vectorstore must be provided")
        if topic is None: raise ValueError("Topic must be provided")
    
    def compile(self):
        # Return the chain
        prompt = PromptTemplate(
            template=self.prompt,
            input_variables=["topic"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        retriever = self.vectorstore.as_retriever()
        
        runner = RunnableParallel(
            {"context": retriever, "topic": RunnablePassthrough()}
        )
        
        chain = runner | prompt | self.model | self.parser
        
        if self.verbose: logger.info(f"Chain compilation complete")
        
        return chain

    def validate_response(self, response: Dict) -> bool:
        try:
            # Assuming the response is already a dictionary
            if isinstance(response, dict):
                if 'question' in response and 'choices' in response and 'answer' in response and 'explanation' in response:
                    choices = response['choices']
                    if isinstance(choices, dict):
                        for key, value in choices.items():
                            if not isinstance(key, str) or not isinstance(value, str):
                                return False
                        return True
            return False
        except TypeError as e:
            if self.verbose:
                logger.error(f"TypeError during response validation: {e}")
            return False

    def format_choices(self, choices: Dict[str, str]) -> List[Dict[str, str]]:
        return [{"key": k, "value": v} for k, v in choices.items()]
    
    def create_questions(self, num_questions: int = 5) -> List[Dict]:
        if self.verbose: logger.info(f"Creating {num_questions} questions")
        
        if num_questions > 10:
            return {"message": "error", "data": "Number of questions cannot exceed 10"}
        
        chain = self.compile()
        
        generated_questions = []
        attempts = 0
        max_attempts = num_questions * 5  # Allow for more attempts to generate questions

        while len(generated_questions) < num_questions and attempts < max_attempts:
            response = chain.invoke(self.topic)
            if self.verbose:
                logger.info(f"Generated response attempt {attempts + 1}: {response}")
            
            # Directly check if the response format is valid
            if self.validate_response(response):
                response["choices"] = self.format_choices(response["choices"])
                generated_questions.append(response)
                if self.verbose:
                    logger.info(f"Valid question added: {response}")
                    logger.info(f"Total generated questions: {len(generated_questions)}")
            else:
                if self.verbose:
                    logger.warning(f"Invalid response format. Attempt {attempts + 1} of {max_attempts}")
            
            # Move to the next attempt regardless of success to ensure progress
            attempts += 1

        # Log if fewer questions are generated
        if len(generated_questions) < num_questions:
            logger.warning(f"Only generated {len(generated_questions)} out of {num_questions} requested questions")
        
        if self.verbose: logger.info(f"Deleting vectorstore")
        self.vectorstore.delete_collection()
        
        # Return the list of questions
        return generated_questions[:num_questions]

class QuestionChoice(BaseModel):
    key: str = Field(description="A unique identifier for the choice using letters A, B, C, D, etc.")
    value: str = Field(description="The text content of the choice")
class QuizQuestion(BaseModel):
    question: str = Field(description="The question text")
    choices: List[QuestionChoice] = Field(description="A list of choices")
    answer: str = Field(description="The correct answer")
    explanation: str = Field(description="An explanation of why the answer is correct")
