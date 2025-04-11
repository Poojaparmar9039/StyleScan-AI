from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('upload_image/',views.upload_image, name='upload_image'),
    path('display/', views.display_image, name='display_image'),
    path('see_image/',views.see_image,name='see_image'),
    
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)