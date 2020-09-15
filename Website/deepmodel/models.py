from django.db import models


# Create your models here.

class DeepModel(models.Model):
    model_file = models.FileField(upload_to='DICOM/', default="")
    tumor_file = models.FileField(upload_to='DICOM/', default="")

