from pathlib import Path
import os
import logging
from langchain.vectorstores.weaviate import Weaviate
from langchain.chains import ChatVectorDBChain
import weaviate
from weaviate.embedded import EmbeddedOptions
from unstructured.partition.pdf import partition_pdf
from dotenv import load_dotenv
import openai

from langchain.vectorstores.weaviate import Weaviate
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import ChatVectorDBChain

logging.basicConfig(level=logging.INFO)

#define a classer: AbstractExtractor
class AbstractExtractor:
    def __init__(self):
        self.current_section = None  # Keep track of the current section being processed
        self.have_extracted_abstract = (
            False  # Keep track of whether the abstract has been extracted
        )
        self.in_abstract_section = (
            False  # Keep track of whether we're inside the Abstract section
        )
        self.texts = []  # Keep track of the extracted abstract text

    def process(self, element):
        if element.category == "Title":
            self.set_section(element.text)

            if self.current_section == "Abstract":
                self.in_abstract_section = True
                return True

            if self.in_abstract_section:
                return False

        if self.in_abstract_section and element.category == "NarrativeText":
            self.consume_abstract_text(element.text)
            return True

        return True

    def set_section(self, text):
        self.current_section = text
        logging.info(f"Current section: {self.current_section}")

    def consume_abstract_text(self, text):
        logging.info(f"Abstract part extracted: {text}")
        self.texts.append(text)

    def consume_elements(self, elements):
        for element in elements:
            should_continue = self.process(element)

            if not should_continue:
                self.have_extracted_abstract = True
                break

        if not self.have_extracted_abstract:
            logging.warning("No abstract found in the given list of objects.")

    def abstract(self):
        return "\n".join(self.texts)


#open_api_key 
load_dotenv(dotenv_path='chatbot.env')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
openai.api_type = "azure"
openai.api_version = "2023-05-15" # API 版本，未来可能会变
openai.api_base = "https://hkust.azure-api.net"

#create a client object
client = weaviate.Client(
    additional_headers={
        "X-Azure-Api-Key": "OPENAI_API_KEY",
        "X-OpenAI-Organization": "https://hkust.azure-api.net"
    },
    embedded_options=EmbeddedOptions(
        additional_env_vars={"OPENAI_APIKEY": os.environ["OPENAI_API_KEY"]}
    )
)
#configure weaviate schema
client.schema.delete_all()

schema = {
    "class": "Document",
    "vectorizer": "text2vec-openai",
    "properties": [
        {
            "name": "source",
            "dataType": ["text"],
        },
        {
            "name": "abstract",
            "dataType": ["text"],
            "moduleConfig": {
                "generative-openai": {
                "resourceName": "azure",
                "deploymentId": "gpt-35-turbo"
                 },
                "text2vec-openai": {"skip": False, "vectorizePropertyName": False}
            },
        },
    ],
    "moduleConfig": {
        "generative-openai": {
            "resourceName": "azure",
            "deploymentId": "gpt-35-turbo"
        },
        "text2vec-openai": {"model": "ada", "modelVersion": "002", "type": "text"},
    },
}


client.schema.create_class(schema)

#import documents
data_folder = "week5-test-chatbot/data"

data_objects = []

for path in Path(data_folder).iterdir():
    if path.suffix != ".pdf":
        continue

    print(f"Processing {path.name}...")

    elements = partition_pdf(filename=path)

    abstract_extractor = AbstractExtractor()
    abstract_extractor.consume_elements(elements)

    data_object = {"source": path.name, "abstract": abstract_extractor.abstract()}

    data_objects.append(data_object)

#import the objects into weaviate
client.batch.configure(batch_size=100)  # Configure batch
with client.batch as batch:
    for data_object in data_objects:
        batch.add_data_object(data_object, "Document")

#to do query
client = weaviate.Client("http://localhost:8080")

vectorstore = Weaviate(client, "Document", "source")

llm = AzureChatOpenAI(
    openai_api_version=openai.api_version,
    deployment_name='gpt-35-turbo',
    openai_api_key=openai.api_key,
    openai_api_type="azure",
    openai_api_base="https://hkust.azure-api.net",
    temperature=0
)

qa = ChatVectorDBChain.from_llm(llm, vectorstore)

chat_history = []

while True:
    query = input("")
    result = qa({"question": query, "chat_history": chat_history})
    print(result["answer"])
    chat_history = [(query, result["answer"])]