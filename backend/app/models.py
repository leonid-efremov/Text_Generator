from django.db import models


class TextGenerator(models.Model):
    init_word = models.CharField(max_length=500)
    max_len = models.IntegerField(null=True)
    text = models.CharField(max_length=10000)

    def __str__(self):
        return self.text
