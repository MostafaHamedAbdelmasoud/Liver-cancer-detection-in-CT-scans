from django import forms
from .models import DeepModel


class DeepModelForm(forms.ModelForm):
    class Meta:
        model = DeepModel
        fields = ('model_file', 'tumor_file')
