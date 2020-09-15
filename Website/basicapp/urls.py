from django.conf.urls import url
from django.urls import path
from basicapp import views
from django.conf import settings
from django.conf.urls.static import static

app_name = 'basicapp'

urlpatterns = [
    path('about/', views.AboutView.as_view(), name='about'),


    path('upload/', views.upload, name='upload'),
    path('patient_list/', views.patient_list, name='patient_list'),


]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
