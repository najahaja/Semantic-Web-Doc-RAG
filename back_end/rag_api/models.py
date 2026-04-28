from django.db import models
import uuid


class Document(models.Model):
    """Represents an ingested PDF document."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    filename = models.CharField(max_length=255)
    file_path = models.CharField(max_length=500)
    total_chunks = models.IntegerField(default=0)
    ingested_at = models.DateTimeField(auto_now_add=True)
    source_type = models.CharField(max_length=10, default='document')

    def __str__(self):
        return self.filename


class WebSource(models.Model):
    """Represents an ingested web URL."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    url = models.URLField(max_length=2000, unique=True)
    title = models.CharField(max_length=500, blank=True)
    total_chunks = models.IntegerField(default=0)
    ingested_at = models.DateTimeField(auto_now_add=True)
    source_type = models.CharField(max_length=10, default='web')

    def __str__(self):
        return self.url


class Chunk(models.Model):
    """Represents a single text chunk stored in the vector DB."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    content = models.TextField()
    source_type = models.CharField(max_length=10)  # 'document' or 'web'
    source_id = models.CharField(max_length=100)   # doc filename or URL
    chunk_index = models.IntegerField(default=0)
    vector_id = models.CharField(max_length=200, blank=True)  # ID in vector DB
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['source_id', 'chunk_index']

    def __str__(self):
        return f"{self.source_type}:{self.source_id}[{self.chunk_index}]"
