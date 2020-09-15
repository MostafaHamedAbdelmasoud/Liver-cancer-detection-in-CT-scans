from django import forms
from django.contrib.auth.forms import UserCreationForm, UserChangeForm
from django.contrib.auth.models import User
from .models import CustomUser

JOBS_CHOICES = [
    ('Doctor', 'Doctor'),
    ('Egineer', 'Egineer'),
    ('Radiologist', 'Radiologist'),

]


class CustomUserCreationForm(UserCreationForm):
    class Meta(UserCreationForm):
        model = CustomUser
        fields = ('email',)


class CustomUserChangeForm(UserChangeForm):
    class Meta:
        model = CustomUser
        fields = ('email',)


class SignUpForm(UserCreationForm):
    class Meta(UserCreationForm.Meta):
        model = CustomUser
        fields = ('username', 'email', 'password1', 'password2', 'specialty')
        help_texts = {
            'username': None,
            'email': None,
            'password1': None,
            'password2': None,
            'specialty': None,
        }


class UserForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput())

    class Meta:
        model = CustomUser
        fields = ('email', 'password')

