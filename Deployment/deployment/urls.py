from django.urls import path
from . import views

urlpatterns = [
    path('tryityourself',views.tryityourself),
    path('summary',views.summary, name='summary'),
    path('about',views.about, name='about'),
    path('architecture',views.architecture, name='architecture'),
    path('result',views.result, name='result'),
    path('askmeanything',views.askmeanything, name='askmeanything')

]