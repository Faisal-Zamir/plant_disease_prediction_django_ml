from django.shortcuts import render

def homepage(request):
    return render(request, 'disease_predictor/homepage.html')