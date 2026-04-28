import re
import numpy as np
from django.conf import settings
from langchain_huggingface import HuggingFaceEmbeddings


class EvaluationService:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def _cosine_similarity(self, vec1, vec2):
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def _embed_long_text(self, text, max_chars=1000):
        """Embed long text by splitting into chunks and returning a list of vectors."""
        chunks = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
        if not chunks:
            return [self.embeddings.embed_query(text)]
        return [self.embeddings.embed_query(c) for c in chunks]

    def _normalize_score(self, score, baseline=0.3):
        """Scale cosine similarity from [baseline, 1.0] to [0.0, 1.0] for more intuitive display."""
        if score <= baseline:
            return 0.0
        # Use a simple linear mapping or sigmoid
        normalized = (score - baseline) / (1.0 - baseline)
        return min(1.0, normalized)

    def _clean_answer(self, text):
        """Strip metadata artifacts from the answer before embedding.
        
        Removes:
        - Timestamp citations like [3.00s - 6.00s]
        - SourceIDs trailing line like 'SourceIDs: [file.mp3]'
        """
        # Remove timestamp patterns e.g. [3.00s - 6.00s]
        text = re.sub(r'\[\d+\.?\d*s\s*-\s*\d+\.?\d*s\]', '', text)
        # Remove SourceIDs line
        text = re.sub(r'SourceIDs:\s*\[.*?\]', '', text, flags=re.IGNORECASE)
        # Collapse extra whitespace
        return ' '.join(text.split()).strip()

    def compute_metrics(self, question, generated_answer, ground_truth=None, context=None):
        """Compute RAG evaluation metrics with optimized comparison logic."""
        
        # Clean the answer of timestamp/metadata noise before evaluating
        clean_answer = self._clean_answer(generated_answer)
        
        # 1. Relevance (Query vs Answer)
        q_vec = self.embeddings.embed_query(question)
        a_vec = self.embeddings.embed_query(clean_answer)
        raw_relevance = self._cosine_similarity(q_vec, a_vec)
        relevance = self._normalize_score(raw_relevance, baseline=0.35)

        # 2. Similarity (Answer vs Ground Truth)
        similarity = 0.0
        if ground_truth:
            gt_vec = self.embeddings.embed_query(ground_truth)
            raw_similarity = self._cosine_similarity(a_vec, gt_vec)
            # Answer vs GT baseline is higher because they should be nearly identical
            similarity = self._normalize_score(raw_similarity, baseline=0.4)

        # 3. Faithfulness: Answer grounded in context
        faithfulness = 0.0
        if context:
            # IMPROVEMENT: Instead of averaging, find the MAX similarity with any chunk
            # This detects if the answer is supported by ANY part of the context.
            context_vecs = self._embed_long_text(context)
            chunk_similarities = [self._cosine_similarity(a_vec, cv) for cv in context_vecs]
            raw_faithfulness = max(chunk_similarities) if chunk_similarities else 0.0
            faithfulness = self._normalize_score(raw_faithfulness, baseline=0.35)
        else:
            faithfulness = relevance * 1.1

        return {
            "relevance": round(float(relevance), 3),
            "faithfulness": round(float(faithfulness), 3),
            "similarity": round(float(similarity), 3)
        }
