from django.contrib import admin
from django.urls import include, path
from . import views

urlpatterns = [
    path('', views.home, name="home"),
    path('sign_out', views.sign_out, name="sign_out")
]