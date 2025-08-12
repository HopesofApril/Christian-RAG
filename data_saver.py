import os
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from data_loader import HWPLoader
import hashlib

# === 설정 ===
EMBEDDING_MODEL = "nomic-embed-text"
VECTORSTORE_DIR = "./vectorstore"
INDEX_NAME = "faiss_index"

# === 임베딩 모델 ===
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)


def build_vectorstore_from_dir(document_dir: str):
    """지정된 폴더의 모든 문서를 읽어 벡터스토어를 생성 또는 업데이트"""
    # 벡터스토어 초기화 (없으면 생성)
    if not os.path.exists(f"{VECTORSTORE_DIR}/{INDEX_NAME}.faiss"):
        if not os.path.exists(VECTORSTORE_DIR):
            os.mkdir(VECTORSTORE_DIR)
        dim = len(embeddings.embed_query("hello world"))
        vectorstore = FAISS(
            embedding_function=embeddings,
            index=faiss.IndexFlatL2(dim),
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
    else:
        vectorstore = FAISS.load_local(
            folder_path=VECTORSTORE_DIR,
            index_name=INDEX_NAME,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )

    # 디렉토리 내 문서 순회
    for filename in os.listdir(document_dir):
        if not filename.lower().endswith(".hwp"):
            continue

        print(f"... 📄 {filename} 진행중 ...")

        file_path = os.path.join(document_dir, filename)
        file_hash = hashlib.md5(open(file_path, 'rb').read()).hexdigest()

        try:
            loader = HWPLoader(file_path)
            docs = loader.load()
        except Exception as e:
            print(f"⚠️  {filename} 처리 중 오류 발생 → 건너뜁니다. 오류: {e}")
            continue  # 다음 파일로 넘어감

        # 문서 로드 및 메타데이터 설정
        # loader = HWPLoader(file_path)
        # docs = loader.load()
        
        for doc in docs:
            doc.metadata["file_hash"] = file_hash
            doc.metadata["file_name"] = filename

        # 문서 분할
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        split_docs = text_splitter.split_documents(docs)

        # 문서 벡터화 및 추가
        vectorstore.add_documents(split_docs, embeddings=embeddings)
        print(f"📄 {filename} 벡터화 및 저장 완료!")

    # 최종 저장
    vectorstore.save_local(folder_path=VECTORSTORE_DIR, index_name=INDEX_NAME)
    print(f"✅ 벡터스토어 생성 완료: {VECTORSTORE_DIR}/{INDEX_NAME}.faiss")


# === 폴더 지정 후 실행 ===
if __name__ == "__main__":
    document_dir = "./2008년"
    build_vectorstore_from_dir(document_dir)
