# Generated by Django 5.2 on 2025-04-09 17:27

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Upload',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('userImage', models.ImageField(default='', upload_to='uploads/photos')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]
