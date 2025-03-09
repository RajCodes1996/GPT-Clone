from django.urls import path
from .views import chatbot_response, chatbot_home
from .views import get_suggestions

urlpatterns = [
    path("", chatbot_home, name="chatbot_home"),
    path("chat/", chatbot_response, name="chatbot_response"),
    path("api/suggestions/", get_suggestions, name="get_suggestions"),
]
