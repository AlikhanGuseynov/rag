import os
import gradio as gr
import openai
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from openai import OpenAI

# 🔐 Укажи свой ключ
os.environ["OPENAI_API_KEY"] = ""  # замените на свой

openai.api_key = os.environ["OPENAI_API_KEY"]

# === Настройки
DOCS_DIR = "docs"
PERSIST_DIR = "storage"
MODEL_NAME = "gpt-3.5-turbo"

# === Создание/загрузка индекса

def load_or_build_index():
    if not os.path.exists(PERSIST_DIR):
        print("🔍 Создание индекса...")
        documents = SimpleDirectoryReader(DOCS_DIR).load_data()
        node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)

        index = VectorStoreIndex.from_documents(
            documents,
            transformations=[node_parser],
        )
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        print("📆 Загрузка индекса...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
    return index

# === Функции работы
index = load_or_build_index()
retriever = index.as_retriever(similarity_top_k=3)


def chat_fn(message, history):
    # Ищем фрагменты документа
    nodes = retriever.retrieve(message)
    max_chars_per_chunk = 800
    context = "\n\n".join([node.node.get_content()[:max_chars_per_chunk] for node in nodes])

    # Запрос к OpenAI напрямую
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Ты эксперт..."},
            {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос:\n{message}"}
        ],
        temperature=0,
    )

    answer = response.choices[0].message.content.strip()
    usage = response.usage
    token_line = (
        f"\n\n🔢 Токены — prompt: {usage.prompt_tokens}, "
        f"ответ: {usage.completion_tokens}, всего: {usage.total_tokens}"
    )
    return answer + token_line

# === Интерфейс
iface = gr.ChatInterface(
    fn=chat_fn,
    title="📄 Вопросы по документу",
    description="Задай вопрос по загруженному документу (docs/)",
    theme="default",
    chatbot=gr.Chatbot(height=400),
    textbox=gr.Textbox(placeholder="Напиши вопрос...", container=False, scale=7),
)

iface.launch()
