from django import forms
from django.contrib import admin

from api.models import Device, Profile, Project


class ProjectAdminForm(forms.ModelForm):
    class Meta:
        model = Project
        fields = '__all__'

    def clean(self):
        cleaned_data = super().clean()
        model_value = cleaned_data.get('model')
        mapped_dataset = Project.MODEL_DATASET_MAP.get(model_value)
        if mapped_dataset is not None:
            cleaned_data['dataset'] = mapped_dataset
        return cleaned_data

class ProjectAdmin(admin.ModelAdmin):
    form = ProjectAdminForm

    fieldsets = (
        ('General', {
            'fields': ('title', 'description', 'status')  # other general fields
        }),
        ('Training Parameters', {
            'classes': ('collapse',),
            'fields': (
                'model', 'dataset', 'dataset_type', 'number_of_rounds', 'number_of_apps',
                'training_mode',
                'number_of_samples', 'number_of_epochs', 'responses_ratio_threshold', 'max_training_time',
                'valid_round_training_threshold', 'power_plugged_only', 'battery_level_threshold', 'seed',
                'current_round', 'use_split_learning', 'use_helper', 'use_knowledge_distillation')
        }),
        ('Privacy Parameters', {
            'classes': ('collapse',),
            'fields': (
                'DP_used', 'epsilon', 'delta')
        }),
    )

    class Media:
        js = ('api/project_admin.js',)

# Register your models here.
admin.site.register(Profile)
admin.site.register(Project, ProjectAdmin)
admin.site.register(Device)
