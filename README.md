# DLaaS-Server — Distributed Learning as a Service

**DLaaS (Distributed Learning as a Service)** is a centralized orchestration backend for privacy-preserving, on-device distributed machine learning in mobile environments. It is built **on top of [FLaaS](http://arxiv.org/abs/2206.10963) (Federated Learning as a Service)** and generalizes it from plain cross-device federated averaging into a configurable platform that combines **federated learning, differential privacy, split learning, helper-offloaded aggregation, and knowledge distillation** under a single project abstraction and REST API.

Like its FLaaS foundation, the service is designed to be hosted on the [Heroku](https://heroku.com/) cloud platform (easily adaptable to others). It exposes a REST API for incoming communication from mobile devices, relies on [Pushwoosh](https://www.pushwoosh.com/) for outgoing push notifications (available as a free [Heroku add-on](https://elements.heroku.com/addons/pushwoosh)), uses the free [Heroku Scheduler add-on](https://elements.heroku.com/addons/scheduler) to drive round management on fixed intervals, and stores model parameters, performance metrics, and static files on Amazon S3. All client devices authenticate using JSON Web Tokens (JWT).

DLaaS is **backwards compatible** with FLaaS: a project that enables none of the extended features behaves exactly like the original federated-averaging coordinator.

DLaaS is built on top of the original FLaaS-Server codebase: [github.com/FLaaSResearch/FLaaS-Server](https://github.com/FLaaSResearch/FLaaS-Server).

For background on the underlying FLaaS system, see:
- [FLaaS: Cross-App On-device Federated Learning in Mobile Environments](http://arxiv.org/abs/2206.10963)
- [FLaaS: Federated Learning as a Service](https://arxiv.org/abs/2011.09359)
- [Demo: FLaaS - Practical Federated Learning as a Service for Mobile Applications](https://dl.acm.org/doi/pdf/10.1145/3508396.3517074)
- [Demo: FLaaS - Enabling Practical Federated Learning on Mobile Environments](https://dl.acm.org/doi/10.1145/3498361.3539693)


## What DLaaS adds on top of FLaaS

| Capability | Project toggle | Where | Summary |
|---|---|---|---|
| Federated averaging (FLaaS baseline) | *(default)* | `api/mlmodel.py`, `api/server_control.py` | Per-device weights summed and averaged across contributing devices. |
| Differential privacy | `DP_used` (`No DP` / `Central DP` / `Local DP`) | `api/mlmodel.py` | Central DP clips per-client updates and adds calibrated Gaussian/Laplace noise; noise scale derived from `(ε, δ)` via Opacus RDP accounting. |
| Split learning | `use_split_learning` | `api/mlmodel.py`, `helper/main.py` | Devices upload frozen-backbone activations + labels; the server/helper trains the classifier head. Dataset-aware (CIFAR-10 and WuW). |
| Helper-offloaded aggregation | `use_helper` | `api/libs/helper_client.py`, `helper/` | Client updates are partitioned and aggregated in parallel by on-demand FastAPI/Docker helper containers, reducing coordinator memory pressure. |
| Knowledge distillation | `use_knowledge_distillation` | `kd/` | Offline teacher→student distillation with annealed temperature and α, as an alternative training mode. |

These features are **composable**: a single project can, for example, run split learning through helpers under central differential privacy.


## Supported models and datasets

DLaaS pairs each model type with a dataset automatically (`Project.MODEL_DATASET_MAP`):

| Project model | Dataset | Per-sample bottleneck | Classes | Trainable head |
|---|---|---|---|---|
| `CIFAR10_B20` | CIFAR-10 | `7×7×1280` = 62 720 (MobileNetV2 backbone, NHWC/TFLite layout) | 10 | `Linear(62720, 10)` |
| `MobileNet` | WuW (wake-word) | `576` (MobileNetV3 + MFCC frontend) | 2 | `Linear(576, 2)` |

Each device receives a flat `model_weights.bin` (the flattened head kernel + bias), trains locally, and returns the updated weights for aggregation.


## Distributed training lifecycle

A `tick` job (`python manage.py tick`, driven by Heroku Scheduler) advances the state machine in `api/scheduling.py`:

1. **Eligibility** — filter devices by power-plugged/battery thresholds, availability heartbeat, and project assignment.
2. **Round creation** — a `Round` copies `number_of_samples`, `number_of_epochs`, and `seed` from the `Project`.
3. **Notification** — eligible devices are notified via Pushwoosh and download the current round model from S3.
4. **Local computation** — clients train (or, in split learning, compute and upload activations) and POST results back through the REST API.
5. **Aggregation** — `server_control.aggregate_model` composes the enabled DP / helper / split-learning paths.
6. **Distribution** — the aggregated model is written to the next round's S3 path; for `MobileNet` projects the deployed `.tflite` is also refreshed from the aggregated head.
7. **Invalid rounds** — if the response ratio is not met before `max_training_time`, the round is marked `INVALID` and the previous model is forwarded unchanged.


## Overview of the REST API

- `/api/get-samples/<:dataset_type>/<:app>/` - Get samples dedicated for specific application and dataset 
- `/api/project/` - List of projects (GET, POST)
- `/api/project/:project_id/` - Details of a project with ID (GET, PUT, DELETE)
- `/api/project/:project_id/get-model/<:round>/` - Download model from project ID and for a given round (GET)
- `/api/project/:project_id/join-round/<:round>/` - Join project with ID and specific round
- `/api/report-availability` - Report Device Availability for project with ID (POST)
- `/api/project/:project_id/submit-results/:round/:filename` - Submit ML evaluation results for project with ID and round (POST)
- `/api/project/:project_id/submit-model/:round/:filename` - Submit local ML model (or split-learning activations) for project with ID and round (POST)
- `/api/token` - Obtain authentication token (for new or rejoining users) (POST)
- `/api/token/refresh` - Refresh authentication token (for existing users) (POST)


## Repository layout

- `api/` — REST endpoints, ORM models (`Project`, `Round`, device request/response), aggregation (`server_control.py`, `mlmodel.py`), scheduler (`scheduling.py`), and DP utilities (Opacus-based accounting).
- `helper/` — FastAPI microservice (Dockerized) for parallel, memory-bounded partial aggregation and helper-side split-learning head training.
- `kd/` — PyTorch knowledge-distillation pipeline (teachers, students, trainer) for CIFAR-10 and WuW.
- `blue/`, `red/` — auxiliary Django apps for project-specific bookkeeping.
- `FLaaS_Server/` — Django project settings, WSGI/ASGI entry points, and routing.


## Configuration

- Create two Amazon S3 storages for storing the model data and static files.

- Create a new heroku app with the following add-ons:
  * Heroku Postgres (or, optionally, use an external PostgreSQL server)
  * Pushwoosh
  * Heroku Scheduler

- Configure the following environmental parameters:
  * AWS_ACCESS_KEY_ID
  * AWS_REGION
  * AWS_S3_BUCKET_NAME
  * AWS_S3_BUCKET_NAME_STATIC
  * AWS_SECRET_ACCESS_KEY
  * PUSHWOOSH_API_TOKEN
  * PUSHWOOSH_APPLICATION_CODE

- Push the repository into Heroku.

- Configure Heroku Scheduler with the following command: `python manage.py tick`.


You should be now able to access and configure the admin interface through `<host>/admin` url.


## Extended features in detail

### Differential privacy
Set per project via `DP_used`:
- **Local DP** — clients perturb updates on-device; the server averages without adding noise.
- **Central DP** — the server clips each client update against the previous global model (`MLModel.dp_accumulate_model`: L2 clipping for Gaussian, L1 for Laplace) and, after averaging, injects calibrated noise (`MLModel.dp_aggregate`). The noise multiplier is computed from the project's `(ε, δ)`, round budget, and sampling rate using the Opacus RDP accountant for Gaussian noise, or a closed-form bound for pure-DP Laplace noise. Aggregated weights are renormalized to a target standard deviation to keep subsequent rounds numerically stable.

### Split learning
Enabled with `use_split_learning`. Instead of training the head locally, a device uploads the frozen backbone's activations (the *bottleneck*) together with one-hot labels for `number_of_samples` examples. The server (or a helper) trains the classifier head from scratch — 10 epochs of full-batch Adam (`lr=1e-3`), Xavier-initialized weights, zero bias — then emits the trained head in TFLite layout so it can be aggregated like any other update. The path is **dataset-aware**:
- **CIFAR-10**: spatial input `[B, 7, 7, 1280]`, permuted to PyTorch NCHW before flattening to `62720`.
- **WuW**: flat input `[B, 1, 576]`, flattened to `576` (no permute).

### Helper-offloaded aggregation
With `use_helper`, client updates are partitioned into groups (default size 5) and dispatched in parallel to on-demand FastAPI helper containers (`helper/main.py`), one per port starting at `8500`. Containers are launched via the Docker SDK and torn down per round. Partial aggregates are mean-combined on the server and can be passed through the central-DP path. The helper protocol carries the project `dataset` so helper-side split learning matches the server, and it records uplink/downlink bytes and peak container memory for communication- and energy-cost analysis.

### Knowledge distillation
The `kd/` package implements offline teacher→student distillation as an alternative training mode (`use_knowledge_distillation`). Pretrained teachers (`teacher_model.pth`, `teacher_model_wuw.pth`) supervise compact students through a combined hard-/soft-label loss with annealed temperature (`T: 4 → 1`) and distillation weight (`α: 0.7 → 0.3`). Both CIFAR-10 and WuW are supported.

### TFLite refresh (MobileNet projects)
After aggregation, `MobileNet` projects reload the Keras checkpoint, blend the aggregated head into the final dense layer via a 0.1/0.9 EWMA, reconvert to TFLite, and persist the artifact so clients fetch an up-to-date `.tflite` while the rest of the network is preserved.


## S3 data layout

```
projects/<project_id>/<round>/model_weights.bin                 ← aggregated round model
projects/<project_id>/<round>/<device_id>/model_weights.bin     ← per-device update / activations
projects/<project_id>/<round>/<device_id>/performance.json      ← per-device metrics
models/<model_name>.bin                                         ← seed model per project type
models/mobilenetv3_mfcc_tf.keras / .tflite                      ← Keras + TFLite refresh artifacts
```


## Server commands

We extended Django's `manage.py` commands to provide server control functionality. The following is a list of the supported commands:

```
python manage.py -h

Type 'manage.py help <subcommand>' for help on a specific subcommand.

Available subcommands:

[api]
    assign
    countresponses
    create-single-user
    createusers
    extract-device-status-responses
    extract-projects-start
    extract-train-responses
    joinedrounds
    performance
    performance_multirounds
    projectresponses
    responses-per-user
    roundstats
    tick
```

In case of heroku instance, use `heroku run` as a prefix in the following commands (or `heroku run bash` to get access to a bash terminal).

### Create user accounts with random passwords.

Command:
`python manage.py createusers 5 --prefix test_user`

Output:
```
username, password, samples_index
test_user1, FUA3ByyLAP, 0
test_user2, M5UJfEjit4, 1
test_user3, uxOlcxrHZi, 2
test_user4, nVJYSxsWEq, 3
test_user5, M5peA7ZzHc, 4
```

Details:
```
Create users accounts with random passwords.

positional arguments:
  accounts              Number of accounts

optional arguments:
  -h, --help            show this help message and exit
  --prefix [PREFIX]     Prefix that will be used in the usernames
  --length [LENGTH]     Password length
  --samples-index-start [SAMPLES_INDEX_START]
                        Sample index start that will increment per user
```


### Assign users to a project

Command:
`python manage.py assign 1 --prefix test_user`

Output:
```
5 user(s) assigned succesfully to project 'Project 1'
```

Details:
```
Query users using a prefix and assign them to a particular project.

positional arguments:
  project               Project ID

optional arguments:
  -h, --help            show this help message and exit
  --prefix [PREFIX]     Prefix of users to be assigned to the given project.
```


### Joined rounds per user

Command:
`python manage.py joinedrounds 1`

Output:
```
username, joined_rounds
test_user1, 16
test_user2, 12
test_user3, 15
```

Details:
```
Report number of joined rounds per user.

positional arguments:
  project               Project ID

optional arguments:
  -h, --help            show this help message and exit
```


### Round performance

Command:
`python manage.py performance 1 --round 1`

Output:
```
Project: Test Baseline IID
Rounds completed: 18/20 (plus 0 invalid)
Round 1 Test Accuracy: TBD
Round 1 Loss: 0.31 (0.01)
```

Details:
```
Report round performance of a project. If a round is not specified, the last one will be used.

positional arguments:
  project               Project ID

optional arguments:
  -h, --help            show this help message and exit
  --round [ROUND]       Round to be evaluated. If not defined, last completed round will be used.
```


### Project responses

Command:
`python manage.py projectresponses 1 --show-details`

Output:
```
Project responses: 4/4 (ratio: 1.00)
    test_user1 - last_reponse: 05/11/2021 16:25:14.761 - version_code: 12 - bucket: 30 - power_plugged: True - battery: 1.00
    test_user2 - last_reponse: 05/11/2021 16:24:55.484 - version_code: 12 - bucket: 30 - power_plugged: True - battery: 1.00
    test_user3 - last_reponse: 05/11/2021 16:24:58.886 - version_code: 12 - bucket: 40 - power_plugged: True - battery: 1.00
    test_user4 - last_reponse: 05/11/2021 16:28:07.406 - version_code: 12 - bucket: 10 - power_plugged: True - battery: 1.00
```

Details:
```
Query and report device responses for users registered to a project.

positional arguments:
  project               Project ID

optional arguments:
  -h, --help            show this help message and exit
  --past-minutes [PAST_MINUTES]
                        Past time in minutes to lookup
  --battery-level [BATTERY_LEVEL]
                        Include power-plugged AND users with >= battery level threshold.
  --plugged-only        Only include power-plugged users
  --show-details        Show details for each user's last response
```


### Responses per user

Command:
`python manage.py responses-per-user`

Output:
```
Filtered Device Status Responses of user 'test_user1':
    05/11/2021 14:47:47.172 - version_code: 12 - bucket: 30 - power_plugged: False - battery: 1.00
    05/11/2021 14:44:04.430 - version_code: 12 - bucket: 30 - power_plugged: True - battery: 1.00
    05/11/2021 14:35:58.184 - version_code: 12 - bucket: 30 - power_plugged: True - battery: 1.00

Filtered Device Status Responses of user 'test_user2':
    05/11/2021 14:43:46.010 - version_code: 12 - bucket: 30 - power_plugged: True - battery: 1.00
    05/11/2021 14:35:40.389 - version_code: 12 - bucket: 10 - power_plugged: True - battery: 1.00

Filtered Device Status Responses of user 'test_user3':
    05/11/2021 14:55:02.405 - version_code: 12 - bucket: 30 - power_plugged: True - battery: 1.00
    05/11/2021 14:49:23.924 - version_code: 12 - bucket: 10 - power_plugged: True - battery: 1.00
    05/11/2021 14:43:49.177 - version_code: 12 - bucket: 10 - power_plugged: True - battery: 1.00
    05/11/2021 14:35:42.836 - version_code: 12 - bucket: 10 - power_plugged: True - battery: 1.00

Filtered Device Status Responses of user 'test_user4':
    05/11/2021 15:12:06.894 - version_code: 12 - bucket: 10 - power_plugged: True - battery: 1.00
    05/11/2021 15:03:10.347 - version_code: 12 - bucket: 10 - power_plugged: True - battery: 1.00
    05/11/2021 14:48:09.845 - version_code: 12 - bucket: 10 - power_plugged: True - battery: 1.00
```

Details:
```
Report device status responses per user.

positional arguments:
  usernames             Usernames to report. If not specified, all registerd users will be reported.

optional arguments:
  -h, --help            show this help message and exit
  --past-minutes [PAST_MINUTES]
                        Past time in minutes to lookup
  --battery-level [BATTERY_LEVEL]
                        Include power-plugged AND users with >= battery level threshold.
  --plugged-only        Only include power-plugged users
```


### Round statistics

Command:
`python manage.py roundstats 1`

Output:
```
round, status, joined_devices_ratio, usernames
0, complete, 0.75, ['test_user2', 'test_user3', 'test_user1']
1, complete, 0.75, ['test_user2', 'test_user3', 'test_user1']
2, complete, 0.75, ['test_user2', 'test_user3', 'test_user1']
3, complete, 0.75, ['test_user2', 'test_user3', 'test_user1']
4, complete, 0.75, ['test_user2', 'test_user3', 'test_user1']
5, complete, 0.75, ['test_user2', 'test_user3', 'test_user1']
6, complete, 0.75, ['test_user2', 'test_user3', 'test_user1']
7, complete, 0.75, ['test_user2', 'test_user3', 'test_user1']
8, complete, 0.75, ['test_user2', 'test_user3', 'test_user1']
9, complete, 0.75, ['test_user2', 'test_user3', 'test_user1']
10, complete, 0.75, ['test_user2', 'test_user3', 'test_user1']
11, complete, 0.75, ['test_user2', 'test_user3', 'test_user1']
12, complete, 0.75, ['test_user2', 'test_user3', 'test_user1']
13, complete, 0.75, ['test_user2', 'test_user3', 'test_user1']
14, complete, 0.75, ['test_user2', 'test_user3', 'test_user1']
15, complete, 0.75, ['test_user2', 'test_user3', 'test_user1']
16, complete, 0.75, ['test_user2', 'test_user3', 'test_user1']
17, complete, 0.75, ['test_user2', 'test_user3', 'test_user1']
18, complete, 1.00, ['test_user2', 'test_user4', 'test_user3', 'test_user1']
19, training, 1.00, ['test_user2', 'test_user4', 'test_user3', 'test_user1']
```

Details:
```
Report joined users per round.

positional arguments:
  project               Project ID

optional arguments:
  -h, --help            show this help message and exit
```
