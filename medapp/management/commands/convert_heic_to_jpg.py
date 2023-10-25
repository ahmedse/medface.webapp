# app/management/commands/convert_heic_to_jpg.py

from django.core.management.base import BaseCommand
import os
from PIL import Image 
from pillow_heif import register_heif_opener

class Command(BaseCommand):
    help = "Converts .heic images to .jpg in a given directory and removes the original .heic files"

    def add_arguments(self, parser):
        parser.add_argument('dir', type=str, help="The directory in which to convert .heic images to .jpg")

    def handle(self, *args, **kwargs):
        register_heif_opener()
        dir_path = kwargs['dir']

        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith(".heic"):
                    heic_path = os.path.join(root, file)
                    image = Image.open(heic_path)

                    jpg_path = os.path.splitext(heic_path)[0] + ".jpg"
                    image.save(jpg_path, format="JPEG")

                    os.remove(heic_path)  # Remove the original .heic file

                    self.stdout.write(self.style.SUCCESS(f'Successfully converted {heic_path} to {jpg_path} and removed the original .heic file'))