from django.db import models



METHOD_CHOICES = (
    ('spiral','Prediction by picture'),
    ('micro', 'Prediction by voice'),
)

class Document(models.Model):
    description = models.CharField(max_length=255, blank=True)
    method = models.CharField(max_length=6, choices=METHOD_CHOICES, default='spiral')
    files = models.FileField(upload_to="files")
    def __str__(self):
            return self.description

class Patient(models.Model):
        name = models.CharField('Name', max_length=50)
        surname = models.CharField('Surname', max_length=50)
        MDVDFo = models.FloatField('MDVP:Fo(Hz)', max_length=50)
        MDVPFhi = models.FloatField('MDVP:Fhi', max_length=50)
        MDVPFlo = models.FloatField('MDVP:Flo', max_length=50)
        MDVPJitterProcent = models.FloatField('MDVP:Jitter(%)', max_length=50)
        MDVPJitterABS = models.FloatField('MDVP:Jitter(Abs)', max_length=50)
        MDVPRAP = models.FloatField('MDVP:RAP', max_length=50)
        MDVPPPQ = models.FloatField('MDVP:PPQ', max_length=50)
        JitterDDP = models.FloatField('Jitter:DDP', max_length=50)
        MDVPShimmer = models.FloatField('MDVP:Shimmer', max_length=50)
        MDVPShimmerdB = models.FloatField('MDVP:Shimmer(dB)', max_length=50)
        ShimmerAPQ3 = models.FloatField('Shimmer:APQ3', max_length=50)
        ShimmerAPQ5 = models.FloatField('Shimmer:APQ5', max_length=50)
        MDVPAPQ = models.FloatField('Shimmer:APQ5', max_length=50)
        ShimmerDDA= models.FloatField('Shimmer:DDA', max_length=50)
        NHR = models.FloatField('NHR', max_length=50)
        HNR = models.FloatField('HNR', max_length=50)
        RPDE = models.FloatField('RPDE', max_length=50)
        DFA = models.FloatField('DFA', max_length=50)
        spread1 = models.FloatField('spread1', max_length=50)
        spread2 = models.FloatField('spread2', max_length=50)
        D2 = models.FloatField('D2', max_length=50)
        PPE = models.FloatField('PPE', max_length=50)

        def __str__(self):
                return self.name + ' ' +self.surname

class spiral_list_Models(models.Model):
        spiral_list=models.DateTimeField(auto_now=True,verbose_name="Name of Model")
        spiral_model_name = models.CharField(max_length=100)

        def __str__(self):
                return self.spiral_model_name
        class Meta:
                verbose_name='spiralModel'
                verbose_name_plural='spiralModels'

class micro_list_Models(models.Model):
        spiral_list = models.DateTimeField(auto_now=True, verbose_name="Name of Model")
        spiral_model_name = models.CharField(max_length=100)

        def __str__(self):
                return self.spiral_model_name

        class Meta:
                verbose_name = 'microModel'
                verbose_name_plural = 'microModels'

class Image(models.Model):
    Name_and_surname=models.CharField(max_length=100)
    image=models.ImageField(upload_to="images")
    def __str__(self):
        return self.Name_and_surname
