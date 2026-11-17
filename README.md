# Vision-Language Model from scratch

A small Vision-Language Model (VLM) that takes an image and a text prompt and generates an answer, built in PyTorch as the group project of the **Advanced AI** course (ModIA, Toulouse INP, 2026). The code for the project lives in Project/; the other `PP*` folders are the course's practical sessions (GPT-2, ViT, CLIP, tokenizer, pretraining, self-supervised learning, deployment).

## Architecture

The model follows the design of small open-source VLMs such as SmolVLM:

- **Vision encoder**: a Vision Transformer loaded with SigLIP2 weights (`google/siglip2-base-patch16-512`), turning a 512×512 image into 1024 patch tokens.
- **Modality projector**: maps image tokens into the language model's embedding space. On top of the baseline projector, we implemented a **C-Abstractor** (inspired by [Honeybee](https://arxiv.org/abs/2312.06742)): convolutional ResBlocks, adaptive average pooling down to 64 tokens, then more ResBlocks, with zero-initialised residuals for stable end-to-end training.
- **Language model**: a decoder-only transformer loaded with SmolLM2-360M-Instruct weights, which generates the answer.

The ViT, language model, projector and training loop were implemented from a skeleton codebase and validated with a unit-test suite.

## Training and evaluation

Training ran on the **Turpan** GPU cluster with SLURM (scripts in `Project/scripts/`), including checkpointing and resumption across job time limits. Models were evaluated on the **MMStar** benchmark (1,500 multiple-choice questions across six categories); our best checkpoint reached **26.1% accuracy**. 
Our accuracy was not better then random due to a small error in code. By mistake, we applied the output projection twice in the attention of the ViT (lines 158 and 159 of Project/models/vision_transformer.py). 

Detailed results are in `Project/eval_results/`.

Despite unsatisfactory results, this project allowed us to better understand the VLM architecture and practice with SLURM on GPU clusters.

## Deployment

The model is served by a **FastAPI** backend and a **Gradio** chat interface, each in its own Docker container:

```bash
cd Project
docker compose up --build
```

The API runs on `localhost:8000` (`/health`, `/chat`) and the chat UI on `localhost:7860`, where you can upload an image and ask questions about it.

## Team

- [Maël Montillet](https://github.com/MaelMontillet)
- [Driss Chraibi](https://github.com/dchrn7)
- [Emeline Castello](https://github.com/emelinecastello)


## About this repository

This is a standalone copy of our team's working repository, [dchrn7/Modern_LLM](https://github.com/dchrn7/Modern_LLM), itself a fork of the course repository. It was copied with full git history (`git clone --bare` + `git push --all`), so every commit keeps its original author.

# Original ReadME

This is the original ReadMe from the original repository of the course.

Welcome to the **Advanced AI** ModIA course page!

This course will give students the basis to understand modern AI systems, with a focus on implementation and applications. The course is structured around:

  - Lectures
  - Practical programming sessions 
  - One project that will start mid-semester.

In this course, you will learn about modern LLMs, VLMs, Self-Supervised Learning, and Reinforcement Learning involved in LLM training. 

This page is the main hub for the course. You will find lecture notes, practical programming and project guidelines, and related code. This is the first year this course is given, so expect some changes and updates along the way. See [below](#repository-setup) for how to update your local repository with the latest changes.


## Repository Setup

First fork this repository to your own GitHub account, then clone it locally

```bash
git clone your_forked_URL
```

Then add this repo as a remote to pull updates

```bash
git remote add upstream https://github.com/paulnovello/Advanced-AI
```

To update your forked repo with the latest changes from this original repo, run:

```bash
git fetch upstream
git merge upstream/main
```

If you only make changes to the files specified in the practical programming sessions, you should never have merge conflict to manaually resolve. If you do, just reach out to the course staff.

## Environment Setup

### 1. Install uv

We recomend to set up the environment once and for all with [uv](https://docs.astral.sh/uv/). Uv is a fast Python package installer and resolver. Install it using one of the following methods:

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Using pip:**
```bash
pip install uv
```

### 2. Create env

```bash
uv env ./aai_env
```


### 3. Install Dependencies

Install the required dependencies using uv:

```bash
uv sync --group student
```

### 4. Activate env

Activate the virtual environment created by uv:

```bash
source ./aai_env/bin/activate  # macOS/Linux
```
or
```bash
.\aai_env\bin\activate    # Windows
```

or use uv's run command to execute scripts within the environment without activating it:

```bash
uv run python # my_script.py
```
