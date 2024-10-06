# text_classifier/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.get_input_text, name='get_input_text'),
]
