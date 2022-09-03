from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from .models import TextGeneratorApp
from main.Text_Generator import TextGenerator


exchange_model = None


def homepage(request):
    params = TextGeneratorApp.objects.all()
    return render(request, 'homepage.html', {"phrase_list": params})


@csrf_exempt
@require_http_methods(["POST", "GET"])
def add(request):
    init_word = request.POST['init_word']  # get from post action in html
    max_len = request.POST['max_len']
    if not isinstance(max_len, int):
        max_len = int(max_len)

    global exchange_model
    if exchange_model is not None:
        model = exchange_model()
    else:
        model = TextGenerator(model_type='GPT')
        train_texts = 'books.csv'
        model.prepare(train_texts, pretrained=True)
        def exchange_model():
            return model

    model.init_word = init_word
    model.max_len = max_len
    text = model.generate()

    phrase = TextGeneratorApp(init_word=init_word, max_len=max_len, text=text)
    phrase.save()
    return redirect('homepage')


def delete(request, phrase_id):
    phrase = TextGeneratorApp.objects.get(id=phrase_id)
    phrase.delete()
    return redirect('homepage')
