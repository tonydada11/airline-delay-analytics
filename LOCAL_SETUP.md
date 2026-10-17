# Local runtime overrides

The committed Compose file uses project `airline-spark` and ports 8080, 8888 and
4040. For an independent checkout set `COMPOSE_PROJECT_NAME` and override ports
in an ignored `docker-compose.override.yml`. Never reuse another checkout's
writable data or artifact mounts. The reconstruction uses project
`airline-history-lab` and ports 8083, 8890 and 4042.

Use `docker compose config` and `docker compose ps` to inspect the effective
runtime. See [runbook](docs/RUNBOOK.md) for supported execution commands.
