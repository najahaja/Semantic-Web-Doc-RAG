import os
from typing import List, Dict, Any, TypedDict
from langgraph.graph import StateGraph,START,END    

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from ..services.vectorstore import VectorStoreService
from ..services.evaluation import EvaluationService


class GraphState(TypedDict):
    question: str
    context: str
    answer: str
    sources: List[Dict[str, str]]
    metrics: Dict[str, float]


class RAGGraph:
    def __init__(self):
        # Initialize Groq LLM
        api_key_groq = os.getenv('GROQ_API_KEY')
        model_name = os.getenv('LLM_MODEL', 'llama-3.1-8b-instant')
        
        self.llm = ChatGroq(
            groq_api_key=api_key_groq,
            model_name=model_name,
            temperature=0
        )
            
        self.vector_store = VectorStoreService()
        self.evaluator = EvaluationService()

    def retrieve(self, state: GraphState) -> Dict[str, Any]:
        """Node 1: Retrieve relevant chunks."""
        from ..models import Chunk as DBChunk
        
        # Increase top_k slightly but we will filter/process them
        results = self.vector_store.retrieve(state["question"], top_k=10)
        
        context_parts = []
        sources = []
        seen_sources = set()
        media_sources_fetched = set()
        
        for doc, score in results:
            source_id = doc.metadata.get("source_id", "Unknown")
            source_type = doc.metadata.get("source_type", "unknown")

            if source_type in ("audio", "video") and source_id not in media_sources_fetched:
                # Limit to top 2 unique media sources to prevent context overflow
                if len(media_sources_fetched) < 2:
                    all_media_chunks = DBChunk.objects.filter(
                        source_id=source_id, source_type=source_type
                    ).order_by("chunk_index")
                    full_transcript = "\n".join(c.content for c in all_media_chunks)
                    context_parts.append(f"Source: {source_id} [Type: {source_type}]\n{full_transcript}")
                    media_sources_fetched.add(source_id)
            elif source_type not in ("audio", "video"):
                context_parts.append(f"Source: {source_id} [Type: {source_type}]\n{doc.page_content}")

            if source_id not in seen_sources:
                sources.append({"id": source_id, "type": source_type})
                seen_sources.add(source_id)
                
        return {
            "context": "\n---\n".join(context_parts),
            "sources": sources
        }

    def rerank(self, state: GraphState) -> Dict[str, Any]:
        """Node 2: Simple rerank/filter (placeholder for complexity)."""
        # In a real app, use a CrossEncoder. 
        # Here we just ensure we have context.
        if not state["context"].strip():
            return {"context": "No relevant information found."}
        return {"context": state["context"]}

    def generate(self, state: GraphState) -> Dict[str, Any]:
        """Node 3: Generate grounded answer."""
        prompt = ChatPromptTemplate.from_template("""
        You are a highly accurate AI assistant answering questions based on the provided Context.
        The Context contains excerpts from documents, websites, or media transcripts.
        
        INSTRUCTIONS:
        1. Answer the Question using ONLY the information in the Context below.
        2. Provide a DIRECT and extremely CONCISE answer. If asked for a definition, provide ONLY the core definition itself without listing qualities, characteristics, or additional details unless explicitly asked.
        3. Do NOT include introductory definitions (if the question isn't asking for one), background context, or additional information. If the question asks for a list, provide ONLY the list items.
        4. If the Context genuinely does not contain any information relevant to the Question, reply EXACTLY: "I don't know based on the provided documents."
        5. Do NOT use external knowledge or make up facts.
        6. SOURCE TYPE RULE (CRITICAL): Each source in the Context is labeled with [Type: audio], [Type: video], or [Type: document]. You MUST read this label carefully and use that EXACT word when referring to the source. If the label says [Type: audio], you MUST say "audio" — NEVER say "video". If the label says [Type: video], you MUST say "video" — NEVER say "audio". Using the wrong type is a critical error.
        7. CITATION RULES:
           - Do NOT mention the source filename (e.g., "According to sample1.pdf") in your answer.
           - For audio/video: You MUST inline-cite the EXACT timestamp(s) from the context (e.g., [12.50s - 15.00s]) WITHIN your answer sentence. This is MANDATORY — do NOT skip timestamps for audio/video answers.
           - After your complete answer, you MUST list the Source IDs used in the format: SourceIDs: [ID1, ID2].
        8. SUMMARIZATION: If specifically asked for a summary of audio or video, provide a cohesive summary WITHOUT timestamps.
        9. Answer the question asked based on ingested pdf or url or audio or video chunks content only, and if there is no relevant information in the context, reply EXACTLY: "I don't know based on the provided documents."
        Context:
        {context}
        
        Question:
        {question}
        
        Answer:
        """
)
        
        chain = prompt | self.llm
        response = chain.invoke({"context": state["context"], "question": state["question"]})
        
        full_response = response.content.strip()
        if "I don't know based on the provided documents" in full_response:
            return {"answer": full_response, "sources": []}
            
        # Parse answer and source IDs
        answer = full_response
        extracted_source_ids = []
        
        if "SourceIDs:" in full_response:
            parts = full_response.split("SourceIDs:")
            answer = parts[0].strip()
            source_part = parts[1].strip().strip("[]")
            extracted_source_ids = [s.strip() for s in source_part.split(",") if s.strip()]

        # Filter sources to only those actually used
        all_sources = state.get("sources", [])
        used_sources = []
        
        if extracted_source_ids:
            # Use the explicitly listed IDs
            extracted_ids_lower = [sid.lower() for sid in extracted_source_ids]
            used_sources = [src for src in all_sources if src["id"].lower() in extracted_ids_lower]
        
        # Fallback if no sources were explicitly listed or parsing failed
        if not used_sources and all_sources:
            if len(all_sources) == 1:
                used_sources = all_sources
            else:
                # Last resort: check if any source ID is mentioned anywhere (just in case)
                for src in all_sources:
                    if src["id"].lower() in full_response.lower():
                        used_sources.append(src)
                
                if not used_sources:
                    used_sources = [all_sources[0]]

        return {"answer": answer, "sources": used_sources}

    def evaluate(self, state: GraphState) -> Dict[str, Any]:
        """Node 4: Evaluate the generated answer."""
        metrics = self.evaluator.compute_metrics(
            state["question"], 
            state["answer"], 
            context=state["context"]
        )
        return {"metrics": metrics}

    def build(self):
        """Compile the LangGraph."""
        workflow = StateGraph(GraphState)
        
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("rerank", self.rerank)
        workflow.add_node("generate", self.generate)
        workflow.add_node("evaluate", self.evaluate)
        
        workflow.add_edge(START,"retrieve")
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "generate")
        workflow.add_edge("generate", "evaluate")
        workflow.add_edge("evaluate", END)
        
        return workflow.compile()

    def run(self, question: str):
        """Execute the pipeline."""
        app = self.build()
        initial_state = {
            "question": question,
            "context": "",
            "answer": "",
            "sources": [],
            "metrics": {}
        }
        return app.invoke(initial_state)
