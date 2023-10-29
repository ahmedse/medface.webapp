# Generated by Django 4.2.6 on 2023-10-29 20:20

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('medapp', '0010_alter_medsessionperson_elected_label'),
    ]

    operations = [
        migrations.CreateModel(
            name='TaskStatus',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('phase', models.CharField(max_length=255)),
                ('status', models.CharField(max_length=255)),
                ('progress', models.IntegerField(default=0)),
                ('last_updated', models.DateTimeField(auto_now=True)),
                ('medsession', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='medapp.medsession')),
            ],
        ),
    ]
