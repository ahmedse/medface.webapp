# Generated by Django 4.2.6 on 2023-11-12 08:15

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('medapp', '0013_hall'),
    ]

    operations = [
        migrations.AlterField(
            model_name='hall',
            name='building',
            field=models.CharField(choices=[('B2', 'B2'), ('B3', 'B3'), ('B4', 'B4'), ('B5', 'B5'), ('B6', 'B6'), ('Hospital', 'Hospital'), ('Other', 'Other')], default='B2', max_length=50, null=True),
        ),
        migrations.AlterField(
            model_name='hall',
            name='floor',
            field=models.CharField(choices=[('0', '0'), ('1', '1'), ('2', '2'), ('3', '3'), ('4', '4')], default='', max_length=50, null=True),
        ),
        migrations.AlterField(
            model_name='hall',
            name='hall_section',
            field=models.CharField(choices=[('PBL', 'PBL'), ('Clinical', 'Clinical'), ('Bedside Teaching', 'Bedside Teaching')], default='', max_length=50, null=True),
        ),
        migrations.AlterField(
            model_name='hall',
            name='notes',
            field=models.CharField(default='', max_length=250, null=True),
        ),
        migrations.AlterField(
            model_name='hall',
            name='type',
            field=models.CharField(choices=[('Lecture Room', 'Lecture Room'), ('Skill Lab', 'Skill Lab'), ('Think Tank', 'Think Tank'), ('Virtual Hospital', 'Virtual Hospital'), ('X-Reality', 'X-Reality'), ('Amphitheatre', 'Amphitheatre'), ('Operating Theatre', 'Operating Theatre')], default='Lecture Room', max_length=50, null=True),
        ),
    ]
