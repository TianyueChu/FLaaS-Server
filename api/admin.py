from django.contrib import admin
from api.models import Profile, Project, Device

class ProjectAdmin(admin.ModelAdmin):
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

# Register your models here.
admin.site.register(Profile)
admin.site.register(Project)
admin.site.register(Device)
