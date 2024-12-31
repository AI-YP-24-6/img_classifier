FROM python:3.12-slim-bookworm

COPY pyproject.toml poetry.lock /workdir/
COPY Backend/ /workdir/Backend
COPY Client/ /workdir/Client
COPY Tools/ /workdir/Tools

WORKDIR /workdir


RUN python -m pip install --no-cache-dir poetry==1.8.5 \
    && poetry config virtualenvs.create false \
    && poetry install --without docs,linters --no-interaction --no-ansi \
    && rm -rf "$(poetry config cache-dir)/\{cache,artifacts\}"

ENV PATH="${PATH}:/root/.local/bin"
ENV PYTHONPATH="/workdir"

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "./Client/app_client.py", "--server.port=8501", "--server.address=0.0.0.0"]
