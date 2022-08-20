from django.urls import path, re_path
from . import views


urlpatterns = [
    path('', views.homepage, name='homepage'),  # , kwargs={"init_word": " ", "max_len": 50}
    path('add', views.add, name='add'),
    path('delete/<int:phrase_id>', views.delete, name='delete'),
    #re_path(r'^$', views.homepage),  # , kwargs={"init_word": " ", "max_len": 50},
]