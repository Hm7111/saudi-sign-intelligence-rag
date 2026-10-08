#!/bin/sh
# Container CMD: start the resident model server in the background, then stay alive for repeated `docker exec`.
python3 /app/server.py >> /tmp/mc3_server.log 2>&1 &
exec sleep infinity
