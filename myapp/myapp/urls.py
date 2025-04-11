from django.contrib import admin
from django.urls import path,include

from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('',include('Home.urls')),
    path('upload/',include('Upload.urls')),
    path('features/',include('Features.urls')),
    path('contact/',include('Contact.urls')),
    path('admin/', admin.site.urls),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
