# SkillLoop HTTP server. Build: docker build -t skillloop .
# Run:   docker run -p 7331:7331 -v skillloop-data:/data -e SKILLLOOP_LOG=json skillloop
FROM python:3.12-slim AS base
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY skillloop ./skillloop
RUN pip install --no-cache-dir ".[openai,mcp]"

# Non-root user; persistent data on a volume
RUN useradd -u 10001 -m skilluser && mkdir -p /data && chown skilluser /data
USER skilluser
ENV SKILLLOOP_HOME=/data \
    SKILLLOOP_LOG=json \
    SKILLLOOP_PORT=7331
VOLUME ["/data"]
EXPOSE 7331

# Health check hits the readiness endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request,os,sys; sys.exit(0 if __import__('json').load(urllib.request.urlopen('http://127.0.0.1:'+os.getenv('SKILLLOOP_PORT','7331')+'/health')).get('ok') else 1)"

CMD ["sh", "-c", "python -m skillloop.cli http --port ${SKILLLOOP_PORT}"]
