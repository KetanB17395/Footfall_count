
from django.urls import path

from . import views

urlpatterns = [
    path('', views.index),
    path('video/', views.video_feed, name='video_feed'),
]