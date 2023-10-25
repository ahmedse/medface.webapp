from django import forms
from .models import Medsession, Person, Image
import datetime
from django.forms import modelformset_factory

class ImageForm(forms.ModelForm):
    class Meta:
        model = Image
        fields = ('image', )

    def clean_image(self):
        image = self.cleaned_data.get('image', False)
        if image:
            if image.size > 10 * 1024 * 1024:
                raise forms.ValidationError('Image file too large ( > 10mb )')
        return image

ImageFormSet = modelformset_factory(Image, form=ImageForm, extra=3, max_num=10)

class MedsessionForm(forms.ModelForm):
    ImageFormSet = modelformset_factory(Image, fields=('image',), extra=3, max_num=10)
    
    PERIOD_CHOICES = [
        ('1', '9-11 AM'),
        ('2', '11-1 PM'),
        ('3', '1-3 PM'),
    ]
    period = forms.ChoiceField(choices=PERIOD_CHOICES, widget=forms.RadioSelect)
    day = forms.DateField(
        initial=datetime.date.today().strftime('%Y-%m-%d'), 
        widget=forms.DateInput(attrs={'type': 'date', 'value': datetime.date.today().strftime('%Y-%m-%d')})
    )
    class Meta:
        model = Medsession
        fields = ['year', 'period', 'day', 'term', 'room', 'course', 'lecturer']

class PersonForm(forms.ModelForm):
    class Meta:
        model = Person
        fields = ['regno', 'fullname', 'year']