# Generated by Django 4.2.6 on 2023-11-20 12:07

from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone
import medapp.models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Medsession',
            fields=[
                ('sessionid', models.AutoField(primary_key=True, serialize=False)),
                ('year', models.CharField(choices=[('1', '1'), ('2', '2'), ('3', '3'), ('4', '4'), ('5', '5'), ('6', '6')], default='3', max_length=1)),
                ('term', models.CharField(choices=[('1', '1'), ('2', '2'), ('3', '3')], default='1', max_length=1)),
                ('day', models.DateField(default=django.utils.timezone.now)),
                ('period', models.CharField(choices=[('1', '1'), ('2', '2'), ('3', '3')], default='1', max_length=1)),
                ('room', models.CharField(max_length=50)),
                ('course', models.CharField(default='n/a', max_length=400)),
                ('lecturer', models.CharField(default='n/a', max_length=400)),
                ('modifytime', models.DateTimeField(auto_now=True)),
            ],
        ),
        migrations.CreateModel(
            name='Person',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('regno', models.CharField(max_length=15, unique=True)),
                ('fullname', models.CharField(max_length=450)),
                ('arname', models.CharField(default='', max_length=450)),
                ('year', models.CharField(choices=[('1', '1'), ('2', '2'), ('3', '3'), ('4', '4'), ('5', '5'), ('6', '6')], default='3', max_length=1)),
                ('year_b', models.CharField(default='22-26', max_length=20)),
                ('group', models.CharField(choices=[('1', '1'), ('2', '2'), ('3', '3'), ('4', '4'), ('5', '5'), ('6', '6'), ('7', '7'), ('8', '8'), ('9', '9'), ('10', '10'), ('11', '11')], max_length=2)),
                ('label', models.CharField(default='', max_length=100)),
                ('modifytime', models.DateTimeField(auto_now=True)),
            ],
        ),
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
        migrations.CreateModel(
            name='TaskStatus',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('status', models.CharField(choices=[('INIT', 'Initializing'), ('PROG', 'In Progress'), ('DONE', 'Done'), ('FAIL', 'Failed')], max_length=255)),
                ('msg', models.CharField(max_length=255)),
                ('progress', models.IntegerField(default=0)),
                ('TimeTaken', models.IntegerField(default=0)),
                ('read', models.BooleanField(default=False)),
                ('last_updated', models.DateTimeField(auto_now=True)),
                ('medsession', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='medapp.medsession')),
            ],
        ),
        migrations.CreateModel(
            name='MedsessionGroup',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('year', models.TextField(null=True)),
                ('group', models.IntegerField(null=True)),
                ('modifytime', models.DateTimeField(auto_now=True)),
                ('medsession', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='medapp.medsession')),
            ],
        ),
        migrations.CreateModel(
            name='Image',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(upload_to=medapp.models.upload_to)),
                ('medsession', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='images', to='medapp.medsession')),
            ],
        ),
        migrations.CreateModel(
            name='MedsessionPerson',
            fields=[
                ('attendance', models.CharField(choices=[('0', 'Not Marked'), ('1', 'Present'), ('2', 'Absent'), ('3', 'Excused Absence'), ('4', 'Tardy/Late'), ('5', 'Early Departure'), ('6', 'Partial Attendance'), ('7', 'Present with Permission')], default='0', max_length=200)),
                ('status', models.CharField(choices=[('1', 'New Added'), ('2', 'By AI'), ('3', 'Manual'), ('4', 'Corrected'), ('5', 'Not Detected'), ('6', 'Unknown ')], default='1', max_length=200)),
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('box', models.TextField(null=True)),
                ('confidence', models.FloatField(null=True)),
                ('image_path', models.TextField(null=True)),
                ('image_url', models.CharField(max_length=200, null=True)),
                ('resnet50_label', models.CharField(max_length=200, null=True)),
                ('resnet50_confidence', models.FloatField(null=True)),
                ('senet50_label', models.CharField(max_length=200, null=True)),
                ('senet50_confidence', models.FloatField(null=True)),
                ('vgg16_label', models.CharField(max_length=200, null=True)),
                ('vgg16_confidence', models.FloatField(null=True)),
                ('elected_label', models.CharField(max_length=200, null=True)),
                ('corrected_label', models.CharField(max_length=200, null=True)),
                ('added_by', models.CharField(choices=[('0', '0'), ('1', '1')], max_length=1)),
                ('medsession', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='medapp.medsession')),
                ('person', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='medapp.person')),
            ],
            options={
                'unique_together': {('medsession', 'person')},
            },
        ),
    ]
