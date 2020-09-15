from django.urls import path
from deepmodel import views

app_name = 'deepmodel'
urlpatterns = [
    path('', views.index, name='index'),
    path('Examine/<int:patient_id>/', views.Examine_img, name='Examine_img'),
    path('slices/', views.get_slice_num, name='get_slice'),
    path('Examine/<int:patient_id>/<int:slice>', views.strt_img, name='start'),
    path('Examine/<int:patient_id>', views.start, name='start_dl'),
    path('slicesdl/<int:patient_id>/<int:slices_num>/<str:message>',
         views.get_slice_dl, name='get_slice_dl'),

    path('UploadModel/', views.UploadModel.as_view(), name='UploadModel')

]
