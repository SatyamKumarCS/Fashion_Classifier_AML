# Python 3.11 is the project's validated version; the image pins it so the host Python doesn't matter.
# Base and uv images are pinned by digest; dependencies come only from the committed uv.lock.
FROM python:3.11-slim@sha256:cce12316475d7d56d605f76e1d811d0dc4e7227f2d4afd29951521264f504ff8
COPY --from=ghcr.io/astral-sh/uv:0.11.7@sha256:240fb85ab0f263ef12f492d8476aa3a2e4e1e333f7d67fbdd923d00a506a516a /uv /uvx /bin/

# The environment lives outside /app so bind-mounting the project (notebook service) can't hide it.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    PATH=/opt/venv/bin:$PATH \
    GRADIO_ANALYTICS_ENABLED=False \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# Linux resolves torch/torchvision to CPU-only wheels (see pyproject.toml); Docker on macOS has no MPS/GPU.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project \
 && python -m ipykernel install --sys-prefix --name aml-fashion --display-name "Python (AML Fashion)"

COPY . .

# Unprivileged runtime user; bind-mounted outputs stay owned by a normal UID on Linux hosts.
RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 7860
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7860"]
