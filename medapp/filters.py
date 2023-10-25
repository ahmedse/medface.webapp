import django_filters
from .models import Medsession

class MedsessionFilter(django_filters.FilterSet):
    class Meta:
        model = Medsession
        fields = ['year', 'term', 'room', 'day']