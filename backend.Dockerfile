FROM python:3.12-slim-bookworm

COPY pyproject.toml poetry.lock /workdir/
COPY Backend/ /workdir/Backend

WORKDIR /workdir

RUN apt-get update \
    && apt-get install -y libgl1-mesa-glx=22.3.6-1+deb12u1 libglib2.0-0=2.74.6-2+deb12u4 --no-install-recommends && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir poetry==1.8.* && \
    poetry config virtualenvs.create false && \
    poetry install --without docs,linters --no-interaction --no-ansi &&\
    rm -rf "$(poetry config cache-dir)/\{cache,artifacts\}"


EXPOSE 54545

HEALTHCHECK CMD curl --fail http://localhost:54545/_stcore/health

# Run the application
CMD ["uvicorn", "Backend.app.main:app", "--host", "0.0.0.0", "--port", "54545"]
