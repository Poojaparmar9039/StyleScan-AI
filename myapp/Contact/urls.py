from django.urls import path
from . import views

urlpatterns = [
    path('contactDetails/',views.contactDetails, name='contactDetails'),
]
