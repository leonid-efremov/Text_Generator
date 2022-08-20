from django.db import models


class TextGeneratorApp(models.Model):
    init_word = models.CharField(max_length=500, default=' ')
    max_len = models.IntegerField(null=True, default=50)
    text = models.CharField(max_length=10000)

    def __str__(self):
        return self.text
