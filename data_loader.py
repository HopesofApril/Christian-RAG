from typing import Any, Dict, List, Optional, Iterator
import olefile
import zlib
import struct
import re
import unicodedata
from langchain.schema import Document
from langchain.document_loaders.base import BaseLoader


class HWPLoader(BaseLoader):
    """HWP 파일 읽기 클래스. HWP 파일의 내용을 문단 단위로 읽습니다."""

    def __init__(self, file_path: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.file_path = file_path
        self.extra_info = {"source": file_path}
        self._initialize_constants()

    def _initialize_constants(self) -> None:
        self.FILE_HEADER_SECTION = "FileHeader"
        self.HWP_SUMMARY_SECTION = "\x05HwpSummaryInformation"
        self.SECTION_NAME_LENGTH = len("Section")
        self.BODYTEXT_SECTION = "BodyText"
        self.HWP_TEXT_TAGS = [67]

    def lazy_load(self) -> Iterator[Document]:
        """HWP 파일에서 문단 단위로 데이터를 읽어 yield 합니다."""
        load_file = olefile.OleFileIO(self.file_path)
        file_dir = load_file.listdir()

        if not self._is_valid_hwp(file_dir):
            raise ValueError("유효하지 않은 HWP 파일입니다.")

        sections = self._get_body_sections(file_dir)

        for section in sections:
            paragraphs = self._get_paragraphs_from_section(load_file, section)
            for para in paragraphs:
                if para.strip():  # 빈 문단 제외
                    yield self._create_document(text=para, extra_info=self.extra_info)

    def _is_valid_hwp(self, dirs: List[List[str]]) -> bool:
        return [self.FILE_HEADER_SECTION] in dirs and [self.HWP_SUMMARY_SECTION] in dirs

    def _get_body_sections(self, dirs: List[List[str]]) -> List[str]:
        section_numbers = [
            int(d[1][self.SECTION_NAME_LENGTH :]) for d in dirs if d[0] == self.BODYTEXT_SECTION
        ]
        return [f"{self.BODYTEXT_SECTION}/Section{num}" for num in sorted(section_numbers)]

    def _create_document(self, text: str, extra_info: Optional[Dict] = None) -> Document:
        return Document(page_content=text, metadata=extra_info or {})

    def _is_compressed(self, load_file: olefile.OleFileIO) -> bool:
        with load_file.openstream(self.FILE_HEADER_SECTION) as header:
            header_data = header.read()
            return bool(header_data[36] & 1)

    def _get_paragraphs_from_section(self, load_file: olefile.OleFileIO, section: str) -> List[str]:
        with load_file.openstream(section) as bodytext:
            data = bodytext.read()

        unpacked_data = zlib.decompress(data, -15) if self._is_compressed(load_file) else data

        i = 0
        text_chunks = []

        while i < len(unpacked_data):
            header_bytes = unpacked_data[i : i + 4]
            header, rec_type, rec_len = self._parse_record_header(header_bytes)

            # 텍스트 레코드만 처리
            if rec_type in self.HWP_TEXT_TAGS:
                rec_data = unpacked_data[i + 4 : i + 4 + rec_len]
                try:
                    text_chunk = rec_data.decode("utf-16")
                    text_chunks.append(text_chunk)
                except UnicodeDecodeError:
                    # 디코딩 실패 시 무시하고 계속 진행
                    pass

            i += 4 + rec_len

        full_text = "".join(text_chunks)
        full_text = self.remove_chinese_characters(full_text)
        full_text = self.remove_control_characters(full_text)

        # 개행 문자로 문단 나누기 + 문단 내 모든 공백 제거
        paragraphs = [re.sub(r"\s+", "", p) for p in full_text.splitlines() if p.strip()]
        return paragraphs

    @staticmethod
    def remove_chinese_characters(s: str) -> str:
        return re.sub(r"[\u4e00-\u9fff]+", "", s)

    @staticmethod
    def remove_control_characters(s: str) -> str:
        return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")

    @staticmethod
    def _parse_record_header(header_bytes: bytes) -> tuple:
        header = struct.unpack_from("<I", header_bytes)[0]
        rec_type = header & 0x3FF
        rec_len = (header >> 20) & 0xFFF
        return header, rec_type, rec_len
    
import json
from typing import Any, Dict, List, Optional, Iterator
from langchain.schema import Document
from langchain.document_loaders.base import BaseLoader


class JsonLoader(BaseLoader):
    """STT 결과 JSON 파일을 읽어 LangChain Document 리스트로 변환합니다."""

    def __init__(self, file_path: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.file_path = file_path
        self.extra_info = {"source": file_path}

    def lazy_load(self) -> Iterator[Document]:
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # segments 안의 text만 Document로 변환
        if "segments" in data:
            for seg in data["segments"]:
                if "text" in seg and seg["text"].strip():
                    yield Document(
                        page_content=seg["text"],
                        metadata={
                            "start": seg.get("start"),
                            "end": seg.get("end"),
                            "confidence": seg.get("confidence"),
                            **self.extra_info,
                        },
                    )
        else:
            # fallback: 전체 JSON을 문자열화
            yield Document(
                page_content=json.dumps(data, ensure_ascii=False, indent=2),
                metadata=self.extra_info,
            )
