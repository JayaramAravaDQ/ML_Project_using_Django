from django.db import models

class MLInputData(models.Model):
    age = models.IntegerField()
    workclass = models.CharField(max_length=100)
    fnlwgt=models.IntegerField()
    education = models.CharField(max_length=100)
    education_num = models.IntegerField()
    marital_status = models.CharField(max_length=100)
    occupation = models.CharField(max_length=100)
    relationship = models.CharField(max_length=100)
    race = models.CharField(max_length=100)
    sex = models.CharField(max_length=100)
    capital_gain = models.IntegerField()
    capital_loss = models.IntegerField()
    hours_per_week = models.IntegerField()
    native_country = models.CharField(max_length=100)

    def __str__(self):
        return f"MLInputData {self.id}"