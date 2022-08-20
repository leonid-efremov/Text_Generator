from django.views.generic import TemplateView
from django.shortcuts import render, redirect
from django.views.decorators.http import require_http_methods
from .models import TextGeneratorApp
from Text_Generator import TextGenerator


exchange_model = None


def homepage(request):
    model = TextGenerator(model_type='GPT')
    train_texts = 'books.csv'
    model.prepare(train_texts, pretrained=True)
    global exchange_model
    def exchange_model():
        return model
    params = TextGeneratorApp.objects.all()
    return render(request, 'homepage.html', {"phrase_list": params})


@require_http_methods(["POST", "GET"])
def add(request):
    model = exchange_model()
    init_word = request.POST['init_word']  # get from post action in html
    max_len = request.POST['max_len']
    text = model.generate() #  'put something here' * (int(max_len)//18)
    phrase = TextGeneratorApp(init_word=init_word, max_len=max_len, text=text)
    phrase.save()
    return redirect('homepage')


def delete(request, phrase_id):
    phrase = TextGeneratorApp.objects.get(id=phrase_id)
    phrase.delete()
    return redirect('homepage')
