from django.urls import path
from . import views

urlpatterns = [
    path('ingest/pdf/', views.ingest_pdf, name='ingest_pdf'),
    path('ingest/url/', views.ingest_url, name='ingest_url'),
    path('ingest/media/', views.ingest_media, name='ingest_media'),
    path('query/', views.query, name='query'),
    path('evaluate/', views.evaluate, name='evaluate'),
    path('reset-db/', views.reset_database, name='reset_db'),
    path('sources/', views.list_sources, name='list_sources'),
]