from django.urls import path
from dev import views
urlpatterns = [
    path("",views.HomeView.as_view()),
    path("endpoint",views.api, name = "api")
]