from django.urls import reverse_lazy, reverse

from .forms import SignUpForm
from .models import CustomUser
from django.views.generic import View, CreateView
from django.contrib.auth import views as auth_views


# Create your views here.
# view for sign up
class SignUpView(CreateView):
    template_name = 'register.html'
    form_class = SignUpForm
    success_url = reverse_lazy('login')


# view for login
class LoginView(auth_views.LoginView):
    template_name = 'login.html'

    def get_success_url(self):
        email = self.request.POST.get('username')
        email = CustomUser.objects.get(email=email)

        if email.specialty == 'doctor':
            return reverse('index')
        else:
            return reverse('deepmodel:UploadModel')
