import warnings
warnings.filterwarnings("ignore")

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

CHROMA_PATH = "db"

PROMPT_TEMPLATE = """
You are a history expert.

Answer using ONLY the given context.

Rules:
- Write EXACTLY 3 lines
- Each line should be a complete sentence
- Do NOT copy raw text
- Remove numbers like 2.6 or headings
- Make answer clear and meaningful

Context:
{context}

Question: {question}

Answer:
"""


def clean_text(text):
    import re
    text = re.sub(r"\d+\.\d+", "", text)  # remove 2.6 type numbers
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():
    # ✅ Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # ✅ DB
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    # ✅ Model
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    print("🤖 RAG Chatbot Ready! (type 'exit' to quit)\n")

    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break

        # 🔥 BEST FIX → more context
        results = db.similarity_search(query, k=5)

        if not results:
            print("\n🤖 Answer:\n No data found.\n")
            continue

        # 🧹 Clean context
        context = " ".join([clean_text(doc.page_content) for doc in results])

        # 🧠 Prompt
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        final_prompt = prompt.format(context=context, question=query)

        # 🤖 Generate
        inputs = tokenizer(final_prompt, return_tensors="pt", truncation=True)

        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True
        )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 🧼 Clean answer
        answer = clean_text(answer)

        print("\n🤖 Answer:\n" + answer + "\n")


if __name__ == "__main__":
    main()
