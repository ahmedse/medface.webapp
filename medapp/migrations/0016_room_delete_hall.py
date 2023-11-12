# Generated by Django 4.2.6 on 2023-11-12 08:22

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('medapp', '0015_alter_hall_building_alter_hall_floor_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='Room',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('room_code', models.CharField(blank=True, max_length=15)),
                ('room_section', models.CharField(blank=True, choices=[('PBL', 'PBL'), ('Clinical', 'Clinical'), ('Bedside Teaching', 'Bedside Teaching')], default='', max_length=50, null=True)),
                ('notes', models.CharField(blank=True, default='', max_length=250, null=True)),
                ('type', models.CharField(blank=True, choices=[('Lecture Room', 'Lecture Room'), ('Skill Lab', 'Skill Lab'), ('Think Tank', 'Think Tank'), ('Virtual Hospital', 'Virtual Hospital'), ('X-Reality', 'X-Reality'), ('Amphitheatre', 'Amphitheatre'), ('Operating Theatre', 'Operating Theatre')], default='Lecture Room', max_length=50, null=True)),
                ('floor', models.CharField(blank=True, choices=[('0', '0'), ('1', '1'), ('2', '2'), ('3', '3'), ('4', '4')], default='', max_length=50, null=True)),
                ('building', models.CharField(blank=True, choices=[('B2', 'B2'), ('B3', 'B3'), ('B4', 'B4'), ('B5', 'B5'), ('B6', 'B6'), ('Hospital', 'Hospital'), ('Other', 'Other')], default='B2', max_length=50, null=True)),
                ('modifytime', models.DateTimeField(auto_now=True)),
            ],
        ),
        migrations.DeleteModel(
            name='Hall',
        ),
    ]
