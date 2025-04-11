from django.shortcuts import render

# Create your views here.
def feature_image(request):
    return render(request, 'Features.html')