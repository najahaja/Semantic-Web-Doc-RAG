import os
from pathlib import Path
from django.conf import settings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LCDocument
from ..models import Chunk, Document, WebSource


class VectorStoreService:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.db_path = Path(settings.VECTOR_DB_PATH)
        self.vector_db = None

    def _load_vector_db(self):
        """Load or create the FAISS index."""
        index_file = self.db_path / "index.faiss"
        if index_file.exists():
            self.vector_db = FAISS.load_local(
                str(self.db_path), 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        else:
            self.vector_db = None
        return self.vector_db

    def _remove_source_from_db(self, source_id, source_type):
        """Delete all existing chunks for a source from both Django DB and FAISS."""
        existing_count = Chunk.objects.filter(source_id=source_id, source_type=source_type).count()
        if existing_count > 0:
            print(f"Removing {existing_count} existing chunks for '{source_id}' before re-ingestion.")
            Chunk.objects.filter(source_id=source_id, source_type=source_type).delete()
            # Rebuild FAISS from the remaining chunks to remove old vectors
            remaining = list(Chunk.objects.all().order_by('source_id', 'chunk_index'))
            if remaining:
                lc_docs = [LCDocument(
                    page_content=c.content,
                    metadata={"source_type": c.source_type, "source_id": c.source_id, "chunk_index": c.chunk_index}
                ) for c in remaining]
                self.vector_db = FAISS.from_documents(lc_docs, self.embeddings)
            else:
                self.vector_db = None
                import shutil
                if self.db_path.exists():
                    shutil.rmtree(self.db_path)
                    return  # Nothing left to save
            self.db_path.mkdir(parents=True, exist_ok=True)
            self.vector_db.save_local(str(self.db_path))

    def add_text(self, text, source_type, source_id):
        """Chunk text, embed, and store in FAISS and Django DB."""
        if not text.strip():
            print("Warning: Received empty text for ingestion.")
            return 0

        # Remove any old chunks for this source to prevent duplicates on re-upload
        self._remove_source_from_db(source_id, source_type)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        texts = splitter.split_text(text)
        
        # Create langchain documents for vector DB
        lc_docs = []
        for i, t in enumerate(texts):
            lc_docs.append(LCDocument(
                page_content=t,
                metadata={
                    "source_type": source_type,
                    "source_id": source_id,
                    "chunk_index": i
                }
            ))

        if not lc_docs:
            return 0

        # Update Vector DB
        if self._load_vector_db() is None:
            self.vector_db = FAISS.from_documents(lc_docs, self.embeddings)
        else:
            self.vector_db.add_documents(lc_docs)
            
        # Save FAISS index
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.vector_db.save_local(str(self.db_path))
        print(f"Vector DB saved to {self.db_path}")

        # Sync with Django DB
        for i, t in enumerate(texts):
            Chunk.objects.create(
                content=t,
                source_type=source_type,
                source_id=source_id,
                chunk_index=i
            )
            
        # Update source stats
        if source_type in ['document', 'video', 'audio']:
            doc, created = Document.objects.get_or_create(filename=source_id)
            doc.source_type = source_type
            doc.total_chunks = len(texts)  # always set absolute count, not accumulate
            doc.save()
        else:
            web, created = WebSource.objects.get_or_create(url=source_id)
            web.total_chunks = len(texts)  # always set absolute count
            web.save()

        return len(texts)

    def add_media_segments(self, segments, source_type, source_id):
        """Chunk media segments by time (5s chunks, 2s overlap), embed, and store in FAISS and Django DB."""
        if not segments:
            print("Warning: Received empty segments for ingestion.")
            return 0

        # Remove any old chunks for this source to prevent duplicates on re-upload
        self._remove_source_from_db(source_id, source_type)
            
        lc_docs = []
        chunk_length = 3.0
        overlap = 1.0
        stride = chunk_length - overlap  # 2.0
        
        duration = segments[-1]['end'] if segments else 0
        chunk_index = 0
        current_time = 0.0
        
        while current_time < duration:
            window_start = current_time
            window_end = current_time + chunk_length
            
            window_texts = []
            for seg in segments:
                # Check if segment overlaps with the current window
                if seg['start'] < window_end and seg['end'] > window_start:
                    window_texts.append(seg['text'].strip())
            
            if window_texts:
                text_content = " ".join(window_texts).strip()
                # Include the timestamp range in the text for the LLM
                chunk_text = f"[{window_start:.2f}s - {window_end:.2f}s] {text_content}"
                
                lc_docs.append(LCDocument(
                    page_content=chunk_text,
                    metadata={
                        "source_type": source_type,
                        "source_id": source_id,
                        "chunk_index": chunk_index,
                        "start_time": window_start,
                        "end_time": window_end
                    }
                ))
                chunk_index += 1
                
            current_time += stride

        if not lc_docs:
            return 0

        # Update Vector DB
        if self._load_vector_db() is None:
            self.vector_db = FAISS.from_documents(lc_docs, self.embeddings)
        else:
            self.vector_db.add_documents(lc_docs)
            
        # Save FAISS index
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.vector_db.save_local(str(self.db_path))
        print(f"Vector DB saved to {self.db_path}")

        # Sync with Django DB
        for i, doc in enumerate(lc_docs):
            Chunk.objects.create(
                content=doc.page_content,
                source_type=source_type,
                source_id=source_id,
                chunk_index=i
            )
            
        # Update source stats (always set absolute count)
        doc, _ = Document.objects.get_or_create(filename=source_id)
        doc.source_type = source_type
        doc.total_chunks = len(lc_docs)
        doc.save()

        return len(lc_docs)

    def retrieve(self, query, top_k=4):
        """Retrieve top k relevant chunks from FAISS."""
        db = self._load_vector_db()
        if db is None:
            return []
        
        results = db.similarity_search_with_score(query, k=top_k)
        return results

    def clear_all(self):
        """Delete all chunks and sources from both Django DB and FAISS."""
        Chunk.objects.all().delete()
        Document.objects.all().delete()
        WebSource.objects.all().delete()
        
        self.vector_db = None
        import shutil
        if self.db_path.exists():
            shutil.rmtree(self.db_path)
            print("Vector database directory removed.")
        print("All database records cleared.")
