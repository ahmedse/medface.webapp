# Generated by Django 4.2.6 on 2023-10-15 11:19

from django.db import migrations, models
import django.db.models.deletion
import medapp.models


class Migration(migrations.Migration):

    dependencies = [
        ('medapp', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Image',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(upload_to=medapp.models.upload_to)),
                ('medsession', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='images', to='medapp.medsession')),
            ],
        ),
    ]
