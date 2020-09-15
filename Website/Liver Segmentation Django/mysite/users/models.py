from django.db import models

from django.db import models
from .managers import CustomUserManager
from django.contrib.auth.models import AbstractBaseUser
from django.contrib.auth.models import PermissionsMixin
from django.utils import timezone
GENDER_MALE = 'doctor'
GENDER_FEMALE = 'engineer'
GENDER_CHOICES = [
    (GENDER_MALE, 'doctor'),
    (GENDER_FEMALE, 'engineer'),
]


# Create your models here.
class CustomUser(AbstractBaseUser, PermissionsMixin):
    username = models.CharField(max_length=200, default=None, null=True)
    email = models.EmailField(unique=True)
    specialty = models.CharField(max_length=200, default=None, null=True, choices=GENDER_CHOICES)

    is_staff = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    date_joined = models.DateTimeField(default=timezone.now)

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []

    objects = CustomUserManager()

    def __str__(self):
        return self.email
