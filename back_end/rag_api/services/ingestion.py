import os
import requests
from pathlib import Path
from bs4 import BeautifulSoup
import pdfplumber
from django.conf import settings
from .vectorstore import VectorStoreService
import os
import shutil
import imageio_ffmpeg
import trafilatura


class IngestionService:
    @staticmethod
    def extract_text_from_pdf(file_path):
        """Extract text and structure (including tables) from a PDF file using pdfplumber."""
        full_text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                # Extract standard text (which pdfplumber naturally formats better than pypdf)
                content = page.extract_text()
                if content:
                    full_text += content + "\n"
                
                # Extract structured tables to avoid missing grid data
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        for row in table:
                            # Clean up cells and format as a readable row
                            cleaned_row = [str(cell).replace("\n", " ").strip() if cell else "" for cell in row]
                            full_text += " | ".join(cleaned_row) + "\n"
                        full_text += "\n"
                        
        return full_text

    @staticmethod
    def scrape_url(url):
        """Scrape meaningful text from a URL using trafilatura."""
        try:
            # Trafilatura is more robust than simple BeautifulSoup for many sites
            downloaded = trafilatura.fetch_url(url)
            
            if downloaded is None:
                # Check why fetch failed (trafilatura doesn't always throw exceptions)
                # We can use requests as a diagnostic check if fetch returns None
                response = requests.get(url, timeout=10, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                })
                if response.status_code == 402:
                    raise Exception("Access Denied: This site has a paywall and requires payment (402).")
                elif response.status_code == 403:
                    raise Exception("Access Denied: Scraping is forbidden by the site (403).")
                elif response.status_code != 200:
                    raise Exception(f"Failed to fetch content (Status: {response.status_code})")
                downloaded = response.text

            text = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
            
            if not text:
                # Fallback to BeautifulSoup if trafilatura extraction fails
                soup = BeautifulSoup(downloaded, 'lxml')
                for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                    element.decompose()
                main_content = soup.find('article') or soup.find('main') or soup.body
                text = main_content.get_text(separator='\n') if main_content else soup.get_text(separator='\n')
                
            # Clean up whitespace
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            return "\n".join(lines)
        except Exception as e:
            raise Exception(f"Failed to scrape URL {url}: {str(e)}")

    @classmethod
    def process_pdf(cls, file_obj, filename):
        """Handle full PDF ingestion flow."""
        # Use pathlib for more robust directory creation on Windows
        media_path = Path(settings.MEDIA_ROOT)
        media_path.mkdir(parents=True, exist_ok=True)
        
        file_path = media_path / filename
        with open(file_path, 'wb+') as destination:
            for chunk in file_obj.chunks():
                destination.write(chunk)
        
        print(f"File saved to {file_path}")
        
        # Extract text
        text = cls.extract_text_from_pdf(file_path)
        
        # Chunk and Store
        vector_store = VectorStoreService()
        chunks_count = vector_store.add_text(text, source_type='document', source_id=filename)
        
        return {
            "filename": filename,
            "chunks": chunks_count,
            "source_type": "document"
        }

    @classmethod
    def process_url(cls, url):
        """Handle full URL ingestion flow."""
        # Scrape text
        text = cls.scrape_url(url)
        
        # Chunk and Store
        vector_store = VectorStoreService()
        chunks_count = vector_store.add_text(text, source_type='web', source_id=url)
        
        return {
            "url": url,
            "chunks": chunks_count,
            "source_type": "web"
        }

    @staticmethod
    def extract_segments_from_media(file_path, is_video=False):
        """Extract transcription segments from an audio or video file using Whisper."""
        audio_path = file_path
        if is_video:
            try:
                from moviepy import VideoFileClip
                clip = VideoFileClip(str(file_path))
                audio_path = str(file_path).replace('.mp4', '.mp3')
                # Write audio file temporarily
                clip.audio.write_audiofile(audio_path, logger=None)
            except Exception as e:
                print(f"Error extracting audio: {e}")
        
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        ffmpeg_dir = os.path.dirname(ffmpeg_exe)
        ffmpeg_copy = os.path.join(ffmpeg_dir, "ffmpeg.exe")
        if not os.path.exists(ffmpeg_copy):
            shutil.copy(ffmpeg_exe, ffmpeg_copy)
            
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
        
        import whisper
        # Use base model for reasonable speed/accuracy trade-off
        model = whisper.load_model("base")
        result = model.transcribe(str(audio_path))
        return result["segments"]

    @classmethod
    def process_media(cls, file_obj, filename, is_video=False):
        """Handle full Audio/Video ingestion flow."""
        media_path = Path(settings.MEDIA_ROOT)
        media_path.mkdir(parents=True, exist_ok=True)
        
        file_path = media_path / filename
        with open(file_path, 'wb+') as destination:
            for chunk in file_obj.chunks():
                destination.write(chunk)
        
        print(f"Media file saved to {file_path}")
        
        # Extract segments
        segments = cls.extract_segments_from_media(file_path, is_video)
        
        # Chunk and Store
        vector_store = VectorStoreService()
        source_type = 'video' if is_video else 'audio'
        chunks_count = vector_store.add_media_segments(segments, source_type=source_type, source_id=filename)
        
        return {
            "filename": filename,
            "chunks": chunks_count,
            "source_type": source_type
        }
