import os

# Integration tests talk to the compose Postgres on host port 5433 unless overridden.
os.environ.setdefault(
    "DATABASE_URL", "postgresql+psycopg://rp_app:rp_app@localhost:5433/researchpilot"
)
