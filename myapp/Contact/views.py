from django.shortcuts import render

# Create your views here.
def contactDetails(request):
    return render(request, 'Contact.html')