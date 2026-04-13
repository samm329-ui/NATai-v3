import json
from pathlib import Path
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.config import (
    LEARNING_DATA_DIR,
    CHATS_DATA_DIR,
    VECTOR_STORE_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

import logging

logger = logging.getLogger("J.A.R.V.I.S")


class VectorStoreService:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        self.vector_store: Optional[FAISS] = None
        self.retriever_cache: dict = {}

    def _load_learning_data(self) -> List[Document]:
        documents = []
        for file_path in sorted(LEARNING_DATA_DIR.glob("*.txt")):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if content:
                    documents.append(
                        Document(
                            page_content=content,
                            metadata={"source": str(file_path.name)},
                        )
                    )
                    logger.info(
                        "[VECTOR] Loaded learning data file: %s", file_path.name
                    )
            except Exception as e:
                logger.warning("Could not load learning data file %s: %s", file_path, e)

        logger.info("[VECTOR] Total learning data files loaded: %d", len(documents))
        return documents

    def _load_chat_history(self) -> List[Document]:
        documents = []
        for file_path in sorted(CHATS_DATA_DIR.glob("*.json")):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    chat_data = json.load(f)
                messages = chat_data.get("messages", [])
                if not isinstance(messages, list):
                    continue

                lines = []
                for msg in messages:
                    role = msg.get("role")
                    content = msg.get("content")
                    if role and content:
                        lines.append(f"{role.capitalize()}: {content}")

                if lines:
                    chat_content = "\n".join(lines)
                    documents.append(
                        Document(
                            page_content=chat_content,
                            metadata={"source": f"chat_{file_path.stem}"},
                        )
                    )

            except Exception as e:
                logger.warning("Could not load chat history file %s: %s", file_path, e)

        logger.info("[VECTOR] Total chat history files loaded: %d", len(documents))
        return documents

    def _create_vector_store(self) -> FAISS:
        all_documents = self._load_learning_data() + self._load_chat_history()

        if not all_documents:
            logger.warning("[VECTOR] No documents to index. Created placeholder index.")
            return FAISS.from_texts(["No data available yet."], self.embeddings)

        chunks = self.text_splitter.split_documents(all_documents)
        logger.info(
            "[VECTOR] Split %d documents into %d chunks (chunk_size=%d, overlap=%d)",
            len(all_documents),
            len(chunks),
            CHUNK_SIZE,
            CHUNK_OVERLAP,
        )

        vector_store = FAISS.from_documents(chunks, self.embeddings)
        logger.info(
            "[VECTOR] FAISS index built successfully with %d vectors", len(chunks)
        )
        return vector_store

    def _save_vector_store(self):
        if self.vector_store:
            try:
                self.vector_store.save_local(str(VECTOR_STORE_DIR))
            except Exception as e:
                logger.error("Failed to save vector store to disk: %s", e)

    def get_retriever(self, k: int = 10):
        if not self.vector_store:
            try:
                self.vector_store = FAISS.load_local(
                    str(VECTOR_STORE_DIR),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                logger.info("[VECTOR] Loaded existing FAISS index from disk.")
            except Exception as e:
                logger.info(
                    "[VECTOR] Could not load FAISS from disk (%s). Creating new index...",
                    e,
                )
                self.vector_store = self._create_vector_store()
                self._save_vector_store()

        if not self.vector_store:
            raise RuntimeError("Vector store not initialized. This should not happen.")

        if k not in self.retriever_cache:
            self.retriever_cache[k] = self.vector_store.as_retriever(
                search_kwargs={"k": k}
            )

        return self.retriever_cache[k]

    def add_documents(self, docs: List[Document]) -> None:
        if not docs or not self.vector_store:
            return
        self.vector_store.add_documents(docs)
        self._invalidate_cache()
        self._save_vector_store()
        logger.info("[VECTOR] Added %d documents to index", len(docs))

    def add_chat_memory(
        self, session_id: str, user_text: str, assistant_text: str
    ) -> None:
        if not user_text and not assistant_text:
            return
        content = f"User: {user_text}\nAssistant: {assistant_text}"
        doc = Document(
            page_content=content, metadata={"source": f"session_{session_id}"}
        )
        self.add_documents([doc])
        logger.info("[VECTOR] Added chat memory for session %s", session_id)

    def add_summary(self, source: str, summary: str) -> None:
        if not summary:
            return
        doc = Document(page_content=summary, metadata={"source": source})
        self.add_documents([doc])
        logger.info("[VECTOR] Added summary for source: %s", source)

    def _invalidate_cache(self) -> None:
        self.retriever_cache.clear()
        logger.debug("[VECTOR] Retriever cache invalidated")
