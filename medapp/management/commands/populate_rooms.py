from django.core.management.base import BaseCommand, CommandError
from medapp.models import Room

# python manage.py populate_Rooms

class Command(BaseCommand):
    help = 'Populate the room table with initial data'

    def handle(self, *args, **options):
        Rooms_data = [
            # B2 Floor 0
            {"floor": "0", "room_code": "005", "type": "Amphitheatre", "building": "B2"},
            {"floor": "0", "room_code": "004", "type": "X-Reality", "building": "B2"},
            {"floor": "0", "room_code": "006", "type": "X-Reality", "building": "B2"},
            {"floor": "0", "room_code": "001", "type": "Lecture Room", "building": "B2"},
            # B2 Floor 1
            {"floor": "1", "room_code": "102", "type": "Lecture Room", "building": "B2"},
            {"floor": "1", "room_code": "103", "type": "Lecture Room", "building": "B2"},
            {"floor": "1", "room_code": "104", "type": "Think Tank", "building": "B2"},
            {"floor": "1", "room_code": "105", "type": "Lecture Room", "building": "B2"},
            {"floor": "1", "room_code": "106", "type": "Lecture Room", "building": "B2"},
            {"floor": "1", "room_code": "107", "type": "Lecture Room", "building": "B2"},
            # B2 Floor 2
            {"floor": "2", "room_code": "201", "type": "Lecture Room", "building": "B2"},
            {"floor": "2", "room_code": "203", "type": "Lecture Room", "building": "B2"},
            {"floor": "2", "room_code": "204", "type": "Skill Lab", "building": "B2"},
            {"floor": "2", "room_code": "205", "type": "Think Tank", "building": "B2"},
            {"floor": "2", "room_code": "206", "type": "Skill Lab", "building": "B2"},
            {"floor": "2", "room_code": "207", "type": "Lecture Room", "building": "B2"},
            # B2 Floor 3
            {"floor": "3", "room_code": "301", "type": "Operating Theatre", "building": "B2"},
            {"floor": "3", "room_code": "302", "type": "Lecture Room", "building": "B2"},
            {"floor": "3", "room_code": "303", "type": "Virtual Hospital", "building": "B2"},
            {"floor": "3", "room_code": "304", "type": "Virtual Hospital", "building": "B2"},
            {"floor": "3", "room_code": "305", "type": "Lecture Room", "building": "B2"},
            # B4 
            {"floor": "1", "room_code": "124", "type": "Lecture Room", "building": "B4"},
            {"floor": "1", "room_code": "125", "type": "Lecture Room", "building": "B4"},
            {"floor": "1", "room_code": "102", "type": "Lecture Room", "building": "B4"},
            {"floor": "3", "room_code": "302", "type": "Lecture Room", "building": "B4"},
            {"floor": "4", "room_code": "402", "type": "Lecture Room", "building": "B4"},
            {"floor": "4", "room_code": "430", "type": "Lecture Room", "building": "B4"},       
        ]

        for Room_data in Rooms_data:
            try:
                room = Room(**Room_data)
                room.full_clean()  # Run Django's built-in model validation
                room.save()
                self.stdout.write(self.style.SUCCESS(f'Successfully created Room {room.room_code}'))
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Failed to create Room {Room_data["room_code"]}: {e}'))