# Generated by Django 4.2.6 on 2023-11-28 08:04

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('medapp', '0002_alter_medsessionperson_added_by'),
    ]

    operations = [
        migrations.AddField(
            model_name='person',
            name='personal_photo_url',
            field=models.URLField(blank=True, null=True),
        ),
    ]