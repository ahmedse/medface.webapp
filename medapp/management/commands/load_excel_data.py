from django.core.management.base import BaseCommand
import pandas as pd
from medapp.models import Person

# python manage.py load_excel_data merged_df.xlsx
class Command(BaseCommand):
    help = 'Load data from Excel file into Person model'

    def add_arguments(self, parser):
        parser.add_argument('file', type=str, help='The path to the Excel file')

    def handle(self, *args, **kwargs):
        file = kwargs['file']
        df = pd.read_excel(file)

        for index, row in df.iterrows():
            Person.objects.create(
                regno=row['regno'],
                fullname=row['fullname'],
                arname=row['arname'],
                year=row['year'],
                year_b=row['year_b'],
                group=row['group'],
                label=row['label'],
            )

        self.stdout.write(self.style.SUCCESS('Data loaded successfully'))