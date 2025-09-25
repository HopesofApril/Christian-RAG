# 라이브러리 호출
import os
import streamlit as st

from langchain_core.messages.chat import ChatMessage
from langchain_core.output_parsers import StrOutputParser

from load_prompt import load_prompt  # 저장된 프롬프트 템플릿 로더
from dotenv import load_dotenv  # 저장된 api key 로더
from langchain_ollama import ChatOllama
import time  # 작업 수행 시 텀을 주기 위한 라이브러리

# 입력한 데이터 DB에 저장하기 위한 라이브러리
from data_loader import HWPLoader, JsonLoader  # 입력 파일 로더
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough

# 해시값 생성을 위한 라이브러리
from hashlib import md5
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore

# ---------------------------
# 시연용 DummyEmbeddings 정의
# ---------------------------
class DummyEmbeddings:
    def embed_query(self, text):
        # FAISS 검색만 사용할 거라 실제 임베딩은 필요 없음
        return [0.0] * 1536  # 기존 nomic 벡터 차원 수와 동일하게 맞춰주세요

# api key 정보 로드
load_dotenv()

# 캐시 디렉토리 생성
for folder in [".cache", ".cache/files", "./model"]:
    if not os.path.exists(folder):
        os.mkdir(folder)
os.environ["TRANSFORMERS_CACHE"] = "./model"
os.environ["HF_HOME"] = "./model"

st.title("소망이 : RAG 기반 말씀 검색 시스템")

# 경고 메세지를 띄우기 위한 빈 영역 생성
warning_msg = st.empty()

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "chain" not in st.session_state:
    st.session_state["chain"] = None

# ---------------------------
# 기존 FAISS 벡터스토어 로드
# ---------------------------
embeddings = DummyEmbeddings()  # 시연용
vectorstore = FAISS.load_local(
    folder_path="./vectorstore",
    index_name="faiss_index",
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)

# 사이드바 버튼
with st.sidebar:
    uploaded_file = st.file_uploader("파일 업로드", type=["hwp", "json"])
    clear_btn = st.button("초기화")

# 메시지 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        with st.chat_message(chat_message.role):
            st.markdown(chat_message.content)

# 메시지 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))

# 파일 업로드 시 retriever 반환 (임베딩 생략)
def embed_file(file):
    file_content = file.read()
    file_hash = md5(file_content).hexdigest()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    warning_msg.warning("⚠️ 시연용: 새 임베딩 생성 생략, 기존 벡터로만 검색합니다")
    time.sleep(2)
    warning_msg.empty()

    retriever = vectorstore.as_retriever()
    return retriever

# 체인 생성
def create_chain(retriever, model_name="EXAONE-3.5"):
    prompt = load_prompt("prompt.yaml", encoding="utf-8")
    if model_name == "EXAONE-3.5":
        llm = ChatOllama(model="EXAONE3.5-Q8:latest", temperature=0)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# ---------------------------
# 파일 업로드 처리
# ---------------------------
if uploaded_file:
    retriever = embed_file(uploaded_file)
    chain = create_chain(retriever, model_name="EXAONE-3.5")
    st.session_state["chain"] = chain
else:
    retriever = vectorstore.as_retriever()
    chain = create_chain(retriever, model_name="EXAONE-3.5")
    st.session_state["chain"] = chain

# 초기화 버튼
if clear_btn:
    st.session_state["messages"] = []
    st.session_state["chain"] = None

# 출력
print_messages()

# 사용자 입력 처리
user_input = st.chat_input("질문을 입력하세요.")
if user_input:
    docs = retriever.invoke(user_input)
    print("🔍 검색된 문서 개수:", len(docs))
    for i, doc in enumerate(docs):
        print(f"[{i}] {doc.metadata.get('file_name')} - {doc.page_content[:100]}...")

    chain = st.session_state["chain"]
    with st.chat_message("user"):
        st.markdown(user_input)

    response = chain.stream(user_input)
    with st.chat_message("assistant"):
        container = st.empty()
        ai_answer = ""
        for token in response:
            ai_answer += token
            container.markdown(ai_answer)

    add_message("user", user_input)
    add_message("assistant", ai_answer)
