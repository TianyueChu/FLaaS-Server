# Generated by Django 4.2.4 on 2024-07-31 14:30

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Device',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('os', models.CharField(choices=[('Android', 'Android'), ('iOS', 'iOS')], default='Android', help_text='OS.', max_length=10, verbose_name='OS')),
                ('model', models.CharField(blank=True, help_text='Model name.', max_length=30, null=True)),
                ('manufacturer', models.CharField(blank=True, help_text='The manufacturer of the device.', max_length=30, null=True)),
                ('brand', models.CharField(blank=True, help_text='The consumer-visible brand.', max_length=30, null=True)),
                ('build_type', models.CharField(blank=True, help_text="The type of build, like 'user' or 'eng' (Android only).", max_length=30, null=True)),
                ('incremental', models.CharField(blank=True, help_text='Incremental (Android only).', max_length=30, null=True)),
                ('os_version', models.CharField(blank=True, help_text='OS version.', max_length=10, null=True, verbose_name='OS version')),
                ('security_patch', models.CharField(blank=True, help_text='Security Patch (Android only).', max_length=15, null=True)),
                ('samples_index', models.PositiveIntegerField(default=0, help_text='Index of sample file to be downloaded.')),
                ('samples_downloaded', models.BooleanField(default=False, help_text='Has the device downloaded the samples?')),
            ],
        ),
        migrations.CreateModel(
            name='NotificationSent',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('create_date', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name='Project',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('create_date', models.DateTimeField(auto_now_add=True)),
                ('title', models.CharField(max_length=30, unique=True)),
                ('description', models.TextField(blank=True, null=True)),
                ('status', models.CharField(choices=[('Stopped', 'Stopped'), ('In Progress', 'In Progress'), ('Completed', 'Completed')], default='Stopped', help_text='Project State.', max_length=30)),
                ('model', models.CharField(choices=[('CIFAR10_B20', 'CIFAR-10 - B20')], default='CIFAR10_B20', help_text='Model type.', max_length=30)),
                ('dataset', models.CharField(choices=[('CIFAR10', 'CIFAR-10')], default='CIFAR10', help_text='Training dataset.', max_length=30)),
                ('dataset_type', models.CharField(choices=[('IID', 'IID'), ('NonIID', 'Non-IID')], max_length=30)),
                ('training_mode', models.CharField(choices=[('BASELINE', 'Baseline'), ('JOINT_SAMPLES', 'Joint Samples'), ('JOINT_MODELS', 'Joint Models')], max_length=30)),
                ('number_of_rounds', models.PositiveIntegerField(default=20, help_text='Max number of succesfull FL rounds until the project is complete.')),
                ('number_of_apps', models.PositiveIntegerField(default=3, help_text='Number of apps per device.')),
                ('number_of_samples', models.PositiveIntegerField(default=150, help_text='Number of samples per app.')),
                ('number_of_epochs', models.PositiveIntegerField(default=20, help_text='Number of epochs per device.')),
                ('DP_used', models.CharField(choices=[('No DP', 'No DP'), ('Central DP', 'Central DP'), ('Local DP', 'Local DP')], default='No DP', help_text='Different Types of DP to be applied', max_length=30)),
                ('epsilon', models.FloatField(default=1.0, help_text='Privacy parameter for differential privacy.')),
                ('delta', models.FloatField(default=1.0, help_text='The sensitivity of the function.')),
                ('responses_ratio_threshold', models.DecimalField(decimal_places=2, default=0.8, help_text='Ratio of valid devices that needs to be fulfilled for running a trainning round.', max_digits=3)),
                ('max_training_time', models.PositiveIntegerField(default=60, help_text='Max training time (in minutes) for a round to wait for incoming training responses.')),
                ('valid_round_training_threshold', models.DecimalField(decimal_places=2, default=0.7, help_text='A round will only be considered as valid if the given ratio of device responses is fulfilled.', max_digits=3)),
                ('power_plugged_only', models.BooleanField(default=True, help_text='Only allow device training when power plugged.')),
                ('battery_level_threshold', models.DecimalField(decimal_places=2, default=0.6, help_text='Only devices greater or equal to this threshold will be considered as valid for training. (not relevant if Power Plugged Only setting is enabled)', max_digits=3)),
                ('seed', models.PositiveIntegerField(blank=True, default=42524235, help_text='Seed to be used when training. Empty for random.', null=True)),
                ('current_round', models.PositiveIntegerField(default=0)),
            ],
        ),
        migrations.CreateModel(
            name='Round',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('create_date', models.DateTimeField(auto_now_add=True)),
                ('round_number', models.PositiveIntegerField()),
                ('status', models.IntegerField(choices=[(1, 'wait'), (2, 'training'), (3, 'complete'), (4, 'invalid')], default=1)),
                ('start_training_date', models.DateTimeField(blank=True, null=True)),
                ('stop_training_date', models.DateTimeField(blank=True, null=True)),
                ('number_of_samples', models.PositiveIntegerField()),
                ('number_of_epochs', models.PositiveIntegerField()),
                ('seed', models.PositiveIntegerField()),
                ('project', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='rounds', to='api.project')),
            ],
        ),
        migrations.CreateModel(
            name='RequestedTrainingRounds',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date_requested', models.DateTimeField(auto_now_add=True)),
                ('device', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='api.device')),
                ('round', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='api.round')),
            ],
        ),
        migrations.CreateModel(
            name='Profile',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('project', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='profiles', to='api.project')),
                ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='profile', to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='JoinedRounds',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('status', models.IntegerField(choices=[(1, 'join'), (2, 'download_model'), (3, 'train'), (4, 'merge_models'), (5, 'submit_results'), (6, 'complete')], default=1)),
                ('date_joined', models.DateTimeField(auto_now_add=True)),
                ('date_last_state', models.DateTimeField(auto_now_add=True)),
                ('device', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='api.device')),
                ('round', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='api.round')),
            ],
        ),
        migrations.CreateModel(
            name='DeviceTrainRequest',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('create_date', models.DateTimeField(auto_now_add=True)),
                ('devices', models.ManyToManyField(related_name='device_train_requests', to='api.device')),
                ('round', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='device_train_request', to='api.round')),
            ],
        ),
        migrations.CreateModel(
            name='DeviceStatusResponse',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('create_date', models.DateTimeField(auto_now_add=True)),
                ('data', models.JSONField(default=dict)),
                ('device', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='device_status_responses', to='api.device')),
                ('device_train_request', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='device_train_responses', to='api.devicetrainrequest')),
            ],
        ),
        migrations.AddField(
            model_name='device',
            name='joined_rounds',
            field=models.ManyToManyField(related_name='joined_devices', through='api.JoinedRounds', to='api.round'),
        ),
        migrations.AddField(
            model_name='device',
            name='profile',
            field=models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='device', to='api.profile'),
        ),
        migrations.AddField(
            model_name='device',
            name='requested_training_rounds',
            field=models.ManyToManyField(related_name='requested_training_devices', through='api.RequestedTrainingRounds', to='api.round'),
        ),
    ]
