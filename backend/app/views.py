from django.shortcuts import render, redirect
from django.views.decorators.http import require_http_methods
from .models import TextGenerator


def homepage(request):
    params = TextGenerator.objects.all()
    return render(request, 'homepage.html', {"phrase_list": params})  # , "max_len": 50


@require_http_methods(["POST"])
def add(request):
    init_word = request.POST['init_word']  # get from post action in html
    max_len = request.POST['max_len']
    text = 'put something here' * (int(max_len)//18)
    phrase = TextGenerator(init_word=init_word, max_len=max_len, text=text)
    phrase.save()
    return redirect('homepage')


def delete(request, phrase_id):
    phrase = TextGenerator.objects.get(id=phrase_id)
    phrase.delete()
    return redirect('homepage')
