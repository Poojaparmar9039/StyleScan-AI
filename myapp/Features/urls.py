from django.urls import path
from . import views

urlpatterns = [
    path('feature_image/',views.feature_image, name='feature_image'),
]
