from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_PATH = "db"
DATA_PATH = "C:/Users/shrik/Desktop/rag-chatbot/data/books/history.md"


def main():
    loader = TextLoader(DATA_PATH, encoding="utf-8")
    documents = loader.load()

    # 🔥 BEST SETTINGS
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # smaller = better answers
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    print(f"Split into {len(chunks)} chunks.")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_PATH
    )

    print("✅ Database created successfully!")


if __name__ == "__main__":
    main()
