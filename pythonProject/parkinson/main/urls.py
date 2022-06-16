from django.conf import settings
from django.urls import path,include
from django.conf.urls.static import static
from . import views
from . import models


urlpatterns = [
    path('', views.index, name= 'home'),
    path('About Us', views.about, name= 'about'),
    path('Test', views.test, name= 'test'),
    path('Support', views.support, name= 'support'),
    path('uploadModel',views.uploadModel, name='uploadModel'),
    path('testMethod',views.testMethod,name='testMethod'),
    path('spiralTest',views.spiralTest,name='spiralTest'),
    path('spiralModels',views.spiralModels,name='spiralModels'),
    path('microModels',views.microModels,name='microModels'),
    path('to_default',views.to_default,name='to_default')

]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)

