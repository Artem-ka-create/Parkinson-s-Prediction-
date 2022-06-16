from django.contrib import admin
from .models import Patient
from .models import spiral_list_Models
from .models import micro_list_Models
from .models import Image
from .models import Document


# Register your models here.
admin.site.register(Patient)
admin.site.register(spiral_list_Models)
admin.site.register(micro_list_Models)
admin.site.register(Image)
admin.site.register(Document)