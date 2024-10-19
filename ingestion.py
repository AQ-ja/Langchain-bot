import os
from dotenv import load_dotenv

load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from firecrawl import FirecrawlApp
from langchain.schema import Document

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def ingest_docs3() -> None:
    app = FirecrawlApp(api_key=os.environ['FIRECRAWL_API_KEY'])
    urls = [
        "https://www.tecuentolapelicula.com/peliculasfk/interstellar.html",
        "https://www.authorea.com/users/170252/articles/375976-ensayo-de-pel%C3%ADcula-interstellar",
        "url3",
        "url4",
        "url5"
    ]

    for url in urls:
        page_content = app.scrape_url(url=url, params={"onlyMainContent": True})
        #print(f"Con la info de: {url}:\n", page_content)

        doc = Document(page_content=str(page_content), metadata={"source": url})

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        docs = text_splitter.split_documents([doc])

        PineconeVectorStore.from_documents(docs, embeddings, index_name="my-data-proyect")


if __name__ == "__main__": ingest_docs3()
