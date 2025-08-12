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


def build_vectorstore_from_dir_optimized(document_dir: str):
    """HWPLoader에서 바로 분할하여 불필요한 chunk 생성 방지"""
    # === 벡터스토어 초기화 ===
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

    # === 문서 처리 ===
    for filename in os.listdir(document_dir):
        if not filename.lower().endswith(".hwp"):
            continue

        print(f"... 📄 {filename} 진행중 ...")
        file_path = os.path.join(document_dir, filename)
        file_hash = hashlib.md5(open(file_path, 'rb').read()).hexdigest()

        try:
            loader = HWPLoader(file_path)
            # HWPLoader에서 바로 문단 단위 Document 리스트를 받는다고 가정
            docs = loader.load(split_by_paragraph=True)  # <-- 여기서 분할
        except Exception as e:
            print(f"⚠️ {filename} 처리 중 오류 발생 → 건너뜁니다. 오류: {e}")
            continue

        # 메타데이터 추가
        for doc in docs:
            doc.metadata["file_hash"] = file_hash
            doc.metadata["file_name"] = filename

        # 후처리 분할 (길이가 너무 긴 문단만 자르기)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=20,
            separators=["\n\n", "\n", " "]
        )
        split_docs = []
        for doc in docs:
            if len(doc.page_content) > 3000:
                split_docs.extend(text_splitter.split_documents([doc]))
            else:
                split_docs.append(doc)

        # 벡터스토어에 추가
        vectorstore.add_documents(split_docs, embeddings=embeddings)
        print(f"📄 {filename} 벡터화 및 저장 완료! (총 {len(split_docs)} chunks)")

    # 최종 저장
    vectorstore.save_local(folder_path=VECTORSTORE_DIR, index_name=INDEX_NAME)
    print(f"✅ 벡터스토어 생성 완료: {VECTORSTORE_DIR}/{INDEX_NAME}.faiss")


# === 폴더 지정 후 실행 ===
if __name__ == "__main__":
    document_dir = "./2007년"
    build_vectorstore_from_dir_optimized(document_dir)
