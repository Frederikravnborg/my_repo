# M10 – Docker (DTU‑MLOps)

## Docker

**Core Module**

> IT WORKS ON MY MACHINE
> THEN WE’LL SHIP YOUR MACHINE
> AND THAT IS HOW DOCKER WAS BORN

---

## Image credit

While the above picture may seem silly at first, it is actually pretty close to how Docker came into existence.

A big part of creating an MLOps pipeline is being able to reproduce it. Reproducibility goes beyond versioning our code with git and using conda environments to keep track of our Python installations. To truly achieve reproducibility, we need to capture system‑level components such as:

* Operating system
* Software dependencies (other than Python packages)

Docker provides this kind of system‑level reproducibility by creating isolated program dependencies. In addition to providing reproducibility, one of the key features of Docker is scalability, which is important when we later discuss deployment. Because Docker ensures system‑level reproducibility, it does not (conceptually) matter whether we try to start our program on a single machine or on 1,000 machines at once.

---

## Docker overview

Docker has three main concepts: **Dockerfile**, **Docker image**, and **Docker container**:

* A **Dockerfile** is a basic text document that contains all the commands a user could call on the command line to run an application. This includes installing dependencies, pulling data from online storage, setting up code, and specifying commands to run (e.g. `python train.py`).

Running (or more correctly, *building*) a Dockerfile will create a **Docker image**. An image is a lightweight, standalone, containerized, executable package of software that includes everything (application code, libraries, tools, dependencies, etc.) necessary to make an application run.

Actually running an image will create a **Docker container**. This means that the same image can be launched multiple times, creating multiple containers.

The exercises today will focus on how to construct the actual Dockerfile, as this is the first step to constructing your own container.

---

## Docker sharing

The whole point of using Docker is that sharing applications becomes much easier. In general, we have two options:

1. After creating the Dockerfile, we can simply commit it to GitHub (it’s just a text file) and then ask other users to build the image themselves.
2. After building the image ourselves, we can upload it to an image registry such as Docker Hub, where others can get our image by running `docker pull`, allowing them to instantaneously run it as a container.

---

## Exercises

In the following exercises, we guide you on how to build a Dockerfile for your MNIST repository that will make the training and prediction a self‑contained application.

Please make sure that you somewhat understand each step and do not just copy the exercise. Also note that you probably need to execute the exercise from an elevated terminal, i.e. with administrative privileges.

The exercises today are only an introduction to Docker and some of the steps are unoptimized from a production perspective. For example, we often want to keep the size of the Docker image as small as possible, which we are not focusing on here.

If you are using VS Code, we recommend installing the VS Code Docker extension for getting an overview of built and running images. The **Dev Containers** extension may also be useful.

---

### 1. Install Docker

How much trouble you need to go through depends on your operating system.

* **Windows & macOS**: Install **Docker Desktop**, which includes a GUI for inspecting images and containers.
* **Windows users** who have not installed WSL yet will need to do so (Docker uses it as a backend).

After installing Docker, restart your laptop.

---

### 2. Verify installation

```bash
docker run hello-world
```

Expected output:

```
Hello from Docker!
This message shows that your installation appears to be working correctly.
```

---

### 3. Pull an image from Docker Hub

```bash
docker pull busybox
```

`busybox` is a very small (1–5 MB) containerized application with essential GNU utilities.

---

### 4. List images

```bash
docker images
```

You should now see the `busybox` image.

---

### 5. Run the image

```bash
docker run busybox
```

Nothing happens because no command was provided.

Try again:

```bash
docker run busybox echo "hello from busybox"
```

---

### 6. Inspect containers

```bash
docker ps
```

What does this command do? What if you add `-a`?

---

### 7. Interactive mode

```bash
docker run -it busybox
```

This allows you to explore the filesystem of the container.

#### Cleanup

Containers leave remnants visible via:

```bash
docker ps -a
```

Remove them using:

```bash
docker rm <container_id>
```

Recommended usage:

```bash
docker run --rm <image>
```

---

### 9. Create a Dockerfile for MNIST

Create a file called `train.dockerfile`. You will create one Dockerfile for training and one for prediction.

---

### 10. Choose a base image

Two options:

**Using pip**

```dockerfile
FROM python:3.12-slim
```

**Using uv** (faster dependency installation)

```dockerfile
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
```

---

### 11. Install essentials

```dockerfile
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*
```

---

### 12. Application‑specific steps

Copy only essential files:

```dockerfile
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY data/ data/
```

Why do we need each of these to run training in our Docker container?

---

### Install dependencies

**Using pip**

```dockerfile
WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir
```

Questions:

* What does `--no-cache-dir` do?
* Why is it important for Docker?

**Using uv**

```dockerfile
WORKDIR /
RUN uv sync --locked --no-cache
```

Questions:

* What does `--no-cache` do?
* What does `--locked` do?

---

### Entrypoint

```dockerfile
ENTRYPOINT ["python", "-u", "src/<project-name>/train.py"]
```

The `-u` flag ensures output is streamed directly to the terminal.

Using uv:

```dockerfile
ENTRYPOINT ["uv", "run", "src/<project-name>/train.py"]
```

---

### 13. Build the image

```bash
docker build -f train.dockerfile . -t train:latest
```

**Mac M1/M2 users**:

```bash
docker build --platform linux/amd64 -f train.dockerfile . -t train:latest
docker run --platform linux/amd64 train:latest
```

---

### 14. Run the image

```bash
docker images
docker run --name experiment1 train:latest
```

You can run multiple containers by giving them different names.

---

### Docker images and disk space

Clean unused resources:

```bash
docker system prune
```

---

### 15. Faster rebuilds with cache mounts

Enable BuildKit (Docker ≥ 23.0 does this by default).

**Using pip**

```dockerfile
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt --no-cache-dir
```

**Using uv**

```dockerfile
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync
```

---

### 16. Files created inside containers

Files created inside containers do not appear on your host machine.

#### a) Copy files manually

```bash
docker cp <container_name>:<container_path>/<file> <local_path>
```

#### b) Mount volumes (recommended)

```bash
docker run --name <container_name> \
  -v %cd%/models:/models \
  train:latest
```

---

### 17. Prediction container

Create `evaluate.dockerfile` that runs:

```
src/<project-name>/evaluate.py
```

Example run:

```bash
docker run --name evaluate --rm \
  -v %cd%/trained_model.pt:/models/trained_model.pt \
  -v %cd%/data/test_images.pt:/test_images.pt \
  -v %cd%/data/test_targets.pt:/test_targets.pt \
  evaluate:latest \
  ../../models/trained_model.pt
```

---

### 18. (Optional) GPU support

#### a) Prerequisites

* Docker Engine
* Nvidia GPU + drivers
* Nvidia Container Toolkit

#### b) Test setup

```bash
docker pull nvidia/cuda:11.0.3-base-ubuntu20.04
docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
```

#### c) PyTorch GPU image

```bash
docker pull nvcr.io/nvidia/pytorch:22.07-py3
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:22.07-py3
```

Inside container:

```python
import torch
print(torch.cuda.is_available())
```

---

### 19. (Optional) Dev Containers

Create `.devcontainer/Dockerfile`:

```dockerfile
FROM python:3.12-slim
```

Or with GPU:

```dockerfile
FROM nvcr.io/nvidia/pytorch:22.07-py3
```

Create `.devcontainer/devcontainer.json`:

**Using pip**

```json
{
  "name": "my_working_env",
  "dockerFile": "Dockerfile",
  "postCreateCommand": "pip install -r requirements.txt"
}
```

**Using uv**

```json
{
  "name": "my_working_env",
  "dockerFile": "Dockerfile",
  "postCreateCommand": "uv sync --locked"
}
```

---

### 20. (Optional) DVC in Docker

Add to Dockerfile:

**Using pip**

```dockerfile
RUN dvc init --no-scm
COPY .dvc/config .dvc/config
COPY *.dvc .dvc/
RUN dvc config core.no_scm true
RUN dvc pull
```

**Using uv**

```dockerfile
RUN uv run dvc init --no-scm
COPY .dvc/config .dvc/config
COPY *.dvc .dvc/
RUN uv run dvc config core.no_scm true
RUN uv run dvc pull
```

Credentials are stored in:

* macOS: `~/Library/Caches`
* Linux: `~/.cache`
* Windows: `{user}/AppData/Local`

Example credential file:

```json
{
  "access_token": "...",
  "client_id": "...",
  "client_secret": "...",
  "refresh_token": "..."
}
```

Add before `dvc pull`:

```dockerfile
COPY <path_to_default.json> default.json
RUN dvc remote modify myremote --local gdrive_service_account_json_file_path default.json
```

---

## Knowledge check

1. **Difference between Docker image and container**
   An image is a static template; a container is a running instance of that image.

2. **Three steps to containerize an application**

   1. Write a Dockerfile
   2. Build an image
   3. Run a container

3. **Advantage of Docker containers**
   Consistent, reproducible environments across machines.

4. **Why layered images matter**
   Only changed layers need rebuilding, improving speed and reuse.

---

This covers the minimum you need to know about Docker. For a deeper dive, see *Docker Cookbook* by Sébastien Goasguen.

January 4, 2026
