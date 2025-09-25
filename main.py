# 라이브러리 호출
import os
import streamlit as st

from langchain_core.messages.chat import ChatMessage
from langchain_core.output_parsers import StrOutputParser

from load_prompt import load_prompt # 저장된 프롬프트 템플릿 로더
from dotenv import load_dotenv # 저장된 api key 로더
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
import time # 작업 수행 시 텀을 주기 위한 라이브러리

# 입력한 데이터 DB에 저장하기 위한 라이브러리
from data_loader import HWPLoader, JsonLoader # 입력 파일 로더
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import OllamaEmbeddings
from embedding_api import RemoteOllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough

# 해시값 생성을 위한 라이브러리
from hashlib import md5
import json
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore

# api key 정보 로드
load_dotenv()

# 캐시 디렉토리 생성 (.폴더명: 숨김폴더)
if not os.path.exists(".cache"):
    os.mkdir(".cache")

# 파일 업로드전용 폴더 생성(임시저장용)
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

# ./cache/ 경로에 다운로드 받도록 설정
if not os.path.exists("./model"):
    os.mkdir("./model")

os.environ["TRANSFORMERS_CACHE"] = "./model"
os.environ["HF_HOME"] = "./model"

st.title("소망이 : RAG 기반 말씀 검색 시스템")

# 경고 메세지를 띄우기 위한 빈 영역 생성
warning_msg = st.empty()

# 채팅이 차곡차곡 쌓여야함 --> 메세지들을 기록하는 기능이 필요
# streamlit은 페이지가 매번 새로고침되는 형태 -> 대화기록을 저장해둬야함! -> 페이지가 새로고침이 일어나도 기록이 남아있어야함
# session_state: 대화기록을 저장하기 위한 용도(세션 상태를 저장)
if "messages" not in st.session_state:  # 처음 한번만 실행하기 위한 코드
    st.session_state["messages"] = []

# 처음 한번만 실행하기 위한 코드 : 아무런 파일을 업로드하지 않았을 때
if "chain" not in st.session_state:
    st.session_state["chain"] = None

# 임베딩 모델 설정
# EMBEDDING_MODEL = "nomic-embed-text"
# embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
embeddings = RemoteOllamaEmbeddings("https://your-ollama-server.com/api/embeddings")

# 저장된 데이터 로드
# 벡터스토어 경로 및 초기화
vectorstore = FAISS.load_local(
    folder_path="./vectorstore",
    index_name="faiss_index",
    embeddings=embeddings,
    allow_dangerous_deserialization=True)

# 사이드바 버튼 생성
with st.sidebar:
    # 파일 업로드
    uploaded_file = st.file_uploader("파일 업로드", type=["hwp","json"])

    # 초기화 버튼 생성
    clear_btn = st.button("초기화")


# 이전 대화기록 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        # st.chat_message(chat_message.role).write(chat_message.content)
        with st.chat_message(chat_message.role):
            st.markdown(chat_message.content)


# 새로운 메세지 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 중복 파일을 비교하기 위해
def check_duplicate(file_hash):
    existing_hashes = {
        doc.metadata.get("file_hash")
        for doc in vectorstore.docstore._dict.values()
        if doc.metadata.get("file_hash") is not None
    }
    return file_hash in existing_hashes


# 파일 업로드 되었을 때:
def embed_file(file):

    # 업로드한 파일을 캐시 디렉토리에 저장
    file_content = file.read()
    file_hash = md5(file_content).hexdigest()

    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    # 만약 기존에 있던 데이터라면 벡터스토어에서 검색
    if check_duplicate(file_hash):
        warning_msg.warning("⚠️이미 업로드된 파일입니다!")
        retriever = vectorstore.as_retriever()
        time.sleep(3)
        warning_msg.empty()  # 경고 메시지 제거
        return retriever # 중복이면 여기서 함수 종료
    
    # 파일 확장자 확인
    ext = os.path.splitext(file.name)[1].lower()

    # 문서 로드(Load Documents)
    try:
        if ext == ".hwp":
            loader = HWPLoader(file_path)
            docs = loader.load()

        elif ext == ".json":
            loader = JsonLoader(file_path)
            docs = loader.load()

        else:
            raise ValueError("지원되지 않는 확장자")

    except Exception as e:
        warning_msg.error("현재 지원되지 않는 형식의 파일입니다...😢")
        time.sleep(3)
        warning_msg.empty()
        return retriever

    warning_msg.warning("...📂문서 처리중입니다...")
    
    # 문서 메타데이터에 파일 해시 추가
    for doc in docs:
        doc.metadata["file_hash"] = file_hash
        doc.metadata["file_name"] = file.name

    # 문서 분할(Split Documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)

    # DB 생성(Create DB) 및 저장
    vectorstore.add_documents(split_documents, embeddings=embeddings)
    vectorstore.save_local(folder_path="./vectorstore", index_name="faiss_index")

    warning_msg.empty()
    # 검색기(Retriever) 생성
    retriever = vectorstore.as_retriever()

    return retriever

# 체인 생성
def create_chain(retriever, model_name="EXAONE-3.5"):
    
    # 프롬프트
    prompt = load_prompt("prompt.yaml", encoding="utf-8")
    
    # 언어모델(LLM)
    if model_name == "EXAONE-3.5":
        llm = ChatOllama(model="EXAONE3.5-Q8:latest", temperature=0)
    
    # 체인 생성
    chain = (
        #{"context": retriever | format_docs, "question": RunnablePassthrough()}
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# 파일이 업로드되었을때:
if uploaded_file:
    # 파일 업로드 후 retriever 생성 (작업시간 오래걸릴 예정)
    retriever = embed_file(uploaded_file)
    chain = create_chain(retriever, model_name="EXAONE-3.5") # selected_model
    st.session_state["chain"] = chain
else:  # 데이터를 입력하지 않은 상태로 입력 처리 ; 기존 벡터스토어에 저장된 데이터 기반 답변 생성
    retriever = vectorstore.as_retriever()
    chain = create_chain(retriever, model_name="EXAONE-3.5")
    st.session_state["chain"] = chain

# 초기화 버튼이 눌리면
if clear_btn:
    st.session_state["messages"] = []  # 빈 리스트 만들기
    st.session_state["chain"] = None

# 호출
print_messages()

## 사용자의 입력받기
user_input = st.chat_input("질문을 입력하세요.")

if user_input:  # 사용자의 입력이 들어오면, (prompt 변수에 입력이 담김)
    docs = retriever.invoke(user_input)  # 검색 결과 확인
    print("🔍 검색된 문서 개수:", len(docs))
    for i, doc in enumerate(docs):
        print(f"[{i}] {doc.metadata.get('file_name')} - {doc.page_content[:100]}...")

    # chain 생성
    chain = st.session_state["chain"]

    # 사용자의 입력
    # st.chat_message("user").write(user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    response = chain.stream(
        user_input
    )  # RunnablePassthrough로 받아오므로 값만 넣어주면됨

    with st.chat_message("assistant"):
        # 빈 컨테이너 만들기 -> 스트리밍 출력(토큰별)
        container = st.empty()
        ai_answer = ""  # 빈 문자열에 이어붙이기
        for token in response:
            ai_answer += token
            container.markdown(ai_answer) # markdown 형식으로 출력

    add_message("user", user_input)
    add_message("assistant", ai_answer)
