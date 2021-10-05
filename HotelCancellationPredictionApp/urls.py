from django.contrib import admin
from django.urls import path, include
from .views import *

urlpatterns = [
    path('', index, name=''),
    path('front_page', front, name='front_page'),
    path('data_collect', data, name='data_collect'),
]
