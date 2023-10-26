# Generated by Django 4.2.6 on 2023-10-14 10:39

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Medsession',
            fields=[
                ('sessionid', models.AutoField(primary_key=True, serialize=False)),
                ('period', models.CharField(choices=[('1', '1'), ('2', '2'), ('3', '3')], default='1', max_length=1)),
                ('day', models.DateField()),
                ('term', models.CharField(choices=[('1', '1'), ('2', '2'), ('3', '3')], default='1', max_length=1)),
                ('year', models.CharField(choices=[('1', '1'), ('2', '2'), ('3', '3'), ('4', '4'), ('5', '5'), ('6', '6')], default='3', max_length=1)),
                ('room', models.CharField(max_length=400)),
                ('course', models.CharField(max_length=400)),
                ('lecturer', models.CharField(max_length=400)),
                ('modifytime', models.DateTimeField(auto_now=True)),
            ],
        ),
        migrations.CreateModel(
            name='Person',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('regno', models.CharField(max_length=15, unique=True)),
                ('fullname', models.CharField(max_length=450)),
                ('year', models.CharField(choices=[('1', '1'), ('2', '2'), ('3', '3'), ('4', '4'), ('5', '5'), ('6', '6')], max_length=1)),
                ('modifytime', models.DateTimeField(auto_now=True)),
            ],
        ),
    ]