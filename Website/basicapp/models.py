from django.db import models
from django.utils import timezone
from django.urls import reverse


# Create your models here.


class Patient(models.Model):
    name = models.CharField(max_length=200)
    PATIENT_DICOM = models.FileField(upload_to='DICOM/', default="")
    MASK_DICOM = models.FileField(upload_to='DICOM/', default="")
    Image = models.ImageField(null=True, blank=True)
    Result = models.IntegerField(null=True)


