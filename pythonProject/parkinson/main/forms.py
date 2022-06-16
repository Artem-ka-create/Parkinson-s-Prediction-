from .models import Patient
from django.forms import ModelForm,TextInput

from .models import *
from django import forms
from .models import Image

class DocumentForm(forms.ModelForm):
    class Meta:
        model = Document
        fields = ("description", "method", "files")

class ImageForm(forms.ModelForm):
    class Meta:
        model=Image
        fields=("Name_and_surname","image")



class PatientForm(ModelForm):
    class Meta:
        model = Patient
        fields = ["name","surname"
            ,"MDVDFo","MDVPFhi","MDVPFlo","MDVPJitterProcent","MDVPJitterABS",'MDVPRAP',
                  "MDVPPPQ","JitterDDP","MDVPShimmer","MDVPShimmerdB","ShimmerAPQ3","ShimmerAPQ5","MDVPAPQ",
                  "ShimmerDDA","NHR","HNR","RPDE","DFA","spread1","spread2","D2","PPE"
                  ]
        widgets = {
            "name": TextInput(attrs={
                'class': 'interview__item__input' ,
                'placeholder': 'put your name'
            }),
            "surname": TextInput(attrs={
                'class': 'interview__item__input',
                'placeholder': 'put your surname'
            }),
            "MDVDFo": TextInput(attrs={
                'class': 'interview__item__input',
                'placeholder': 'put MDVDFo'
            }),
            "MDVPFhi": TextInput(attrs={
                'class': 'interview__item__input',
                'placeholder': 'put MDVPFhi'
            }),
            "MDVPFlo": TextInput(attrs={
                'class': 'interview__item__input',
                'placeholder': 'put MDVPFlo'
            }),
            "MDVPJitterProcent": TextInput(attrs={
                'class': 'interview__item__input',
                'placeholder': 'put MDVPJitterProcent'
            }),
            "MDVPJitterABS": TextInput(attrs={
                'class': 'interview__item__input',
                'placeholder': 'put MDVPJitterABS'
            }),
            "MDVPRAP": TextInput(attrs={
                'class': 'interview__item__input',
                'placeholder': 'put MDVPRAP'
            }),
            "MDVPPPQ": TextInput(attrs={
                'class': 'interview__item__input',
                'placeholder': 'put MDVPPPQ'
            }),
            "JitterDDP": TextInput(attrs={
                'class': 'interview__item__input',
                'placeholder': 'put JitterDDP'
            }),
            "MDVPShimmer": TextInput(attrs={
                'class': 'interview__item__input',
                'placeholder': 'put MDVPShimmer'
            }),
            "MDVPShimmerdB": TextInput(attrs={
                'class': 'interview__item__input',
                'placeholder': 'put MDVPShimmerdB'
            }),
            "ShimmerAPQ3": TextInput(attrs={
                'class': 'interview__item__input',
                'placeholder': 'put ShimmerAPQ3'
            }),
            "ShimmerAPQ5": TextInput(attrs={
                'class': 'interview__item__input',
                'placeholder': 'put ShimmerAPQ5'
            }),
            "MDVPAPQ": TextInput(attrs={
                'class': 'interview__item__input',
                'placeholder': 'put MDVPAPQ'
            }),
            "ShimmerDDA": TextInput(attrs={
                'class': 'interview__item__input',
                'placeholder': 'put ShimmerDDA'
            }),
            "NHR": TextInput(attrs={
                'class': 'interview__item__input',
                'placeholder': 'put NHR'
            }),
            "HNR": TextInput(attrs={
                'class': 'interview__item__input',
                'placeholder': 'put HNR'
            }),
            "RPDE": TextInput(attrs={
                'class': 'interview__item__input',
                'placeholder': 'put RPDE'
            }),
            "DFA": TextInput(attrs={
                'class': 'interview__item__input',
                'placeholder': 'put DFA'
            }),
            "spread1": TextInput(attrs={
                'class': 'interview__item__input',
                'placeholder': 'put spread1'
            }),
            "spread2": TextInput(attrs={
                'class': 'interview__item__input',
                'placeholder': 'put spread2'
            }),
            "D2": TextInput(attrs={
                'class': 'interview__item__input',
                'placeholder': 'put D2'}),
            "PPE": TextInput(attrs={
                'class': 'interview__item__input',
                'placeholder': 'put PPE'
            }),

        }



