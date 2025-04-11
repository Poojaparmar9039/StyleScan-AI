from django.db import models

class Upload(models.Model):
    userImage = models.ImageField(upload_to='uploads/',default='')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Upload {self.id}"