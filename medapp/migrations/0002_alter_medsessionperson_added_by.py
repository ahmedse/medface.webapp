# Generated by Django 4.2.6 on 2023-11-22 08:13

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('medapp', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='medsessionperson',
            name='added_by',
            field=models.CharField(choices=[('0', 'By AI'), ('1', 'Manual')], default='0', max_length=1),
        ),
    ]
