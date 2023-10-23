import streamlit as st  # Importa la biblioteca Streamlit para crear la aplicación web
import os

from PyPDF2 import PdfReader  # Importa PyPDF2 para trabajar con documentos PDF
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Importa una función de división de texto personalizada
from langchain.embeddings import HuggingFaceEmbeddings  # Importa HuggingFaceEmbeddings para representaciones de texto
from langchain.vectorstores import FAISS  # Importa FAISS para almacenar vectores y realizar búsquedas de similitud
from langchain.chat_models import ChatOpenAI  # Importa ChatOpenAI para la generación de respuestas de lenguaje natural
from langchain.chains.question_answering import load_qa_chain  # Importa una cadena de procesamiento de preguntas y respuestas

st.set_page_config('preguntaDOC')  # Configura el título de la página web

st.header("Pregunta a tu PDF")  # Muestra un encabezado en la página web

OPENAI_API_KEY = st.text_input('OpenAI API Key', type='password')  # Campo de entrada para la clave de API de OpenAI

pdf_obj = st.file_uploader("Carga tu documento", type="pdf", on_change=st.cache_resource.clear)
# Campo de carga de archivos PDF. Se borra la caché de recursos en cada cambio.

@st.cache_resource
def create_embeddings(pdf):
    # Función para procesar el PDF y crear representaciones incrustadas
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    return knowledge_base

if pdf_obj:
    knowledge_base = create_embeddings(pdf_obj)
    user_question = st.text_input("Haz una pregunta sobre tu PDF:")
    # Campo de entrada de texto para preguntas

    if user_question:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        docs = knowledge_base.similarity_search(user_question, 3)
        llm = ChatOpenAI(model_name='gpt-3.5-turbo')
        chain = load_qa_chain(llm, chain_type="stuff")
        respuesta = chain.run(input_documents=docs, question=user_question)
        # Realiza una búsqueda de similitud, configura un modelo de lenguaje y genera una respuesta

        st.write(respuesta)  # Muestra la respuesta en la página web
