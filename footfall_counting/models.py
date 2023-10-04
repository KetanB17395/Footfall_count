from django.db import models

# Create your models here.
class CountData(models.Model):
    in_count = models.IntegerField(default=0)
    out_count = models.IntegerField(default=0)
    timestamp = models.DateTimeField(auto_now_add=True)  # This will automatically set the timestamp when a record is created
    