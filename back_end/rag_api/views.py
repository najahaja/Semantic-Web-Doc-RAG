from rest_framework.decorators import api_view, parser_classes
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, JSONParser
from .services.ingestion import IngestionService
from .services.evaluation import EvaluationService
from .pipeline.graph import RAGGraph


@api_view(['POST'])
@parser_classes([MultiPartParser])
def ingest_pdf(request):
    """API to upload and ingest a PDF file."""
    file_obj = request.FILES.get('file')
    if not file_obj:
        return Response({"error": "No file uploaded"}, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        result = IngestionService.process_pdf(file_obj, file_obj.name)
        return Response(result, status=status.HTTP_201_CREATED)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@parser_classes([JSONParser])
def ingest_url(request):
    """API to scrape and ingest a web URL."""
    url = request.data.get('url')
    if not url:
        return Response({"error": "No URL provided"}, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        result = IngestionService.process_url(url)
        return Response(result, status=status.HTTP_201_CREATED)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@parser_classes([MultiPartParser])
def ingest_media(request):
    """API to upload and ingest an audio or video file."""
    file_obj = request.FILES.get('file')
    if not file_obj:
        return Response({"error": "No file uploaded"}, status=status.HTTP_400_BAD_REQUEST)
    
    # Explicitly define which extensions are video vs audio
    VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')
    AUDIO_EXTENSIONS = ('.mp3', '.wav', '.ogg', '.m4a', '.flac', '.aac')
    fname_lower = file_obj.name.lower()
    is_video = fname_lower.endswith(VIDEO_EXTENSIONS)
    is_audio = fname_lower.endswith(AUDIO_EXTENSIONS)
    
    if not is_video and not is_audio:
        return Response({"error": "Unsupported file type. Use mp3, wav, ogg, m4a, flac, aac for audio or mp4, avi, mov, mkv for video."}, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        result = IngestionService.process_media(file_obj, file_obj.name, is_video=is_video)
        return Response(result, status=status.HTTP_201_CREATED)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@parser_classes([JSONParser])
def query(request):
    """API to run the RAG pipeline for a question."""
    question = request.data.get('question')
    if not question:
        return Response({"error": "No question provided"}, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        graph = RAGGraph()
        result = graph.run(question)
        return Response({
            "answer": result["answer"],
            "sources": result["sources"],
            "metrics": result["metrics"]
        }, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
def reset_database(request):
    """API to clear all ingested data."""
    try:
        from .services.vectorstore import VectorStoreService
        vs = VectorStoreService()
        vs.clear_all()
        return Response({"message": "Database reset successfully"}, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@parser_classes([JSONParser])
def evaluate(request):
    """API to evaluate a generated answer against a ground truth."""
    question = request.data.get('question')
    answer = request.data.get('answer')
    ground_truth = request.data.get('ground_truth')
    
    if not all([question, answer]):
        return Response({"error": "Missing question or answer"}, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        evaluator = EvaluationService()
        metrics = evaluator.compute_metrics(question, answer, ground_truth)
        return Response(metrics, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def list_sources(request):
    """API to list all ingested documents and URLs."""
    try:
        from .models import Document, WebSource
        docs = Document.objects.all().values('filename', 'source_type', 'ingested_at')
        webs = WebSource.objects.all().values('url', 'source_type', 'ingested_at')
        
        sources = []
        for d in docs:
            sources.append({"id": d['filename'], "type": d['source_type'], "date": d['ingested_at']})
        for w in webs:
            sources.append({"id": w['url'], "type": w['source_type'], "date": w['ingested_at']})
            
        return Response({"sources": sources}, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)