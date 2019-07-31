from django.shortcuts import render
from .models import *
from .word2vec import top3_words

# Create your views here.

def home(request):
	return render(request, 'mykeyword/home.html')
def about(request):
	return render(request, 'mykeyword/about.html')
def count(request):
    full_text = request.GET['fulltext']
    top_words = top3_words(full_text)

    return render(request, 'mykeyword/count.html', {'fulltext': full_text, 'dictionary':top_words.items()})