#!/usr/bin/env bash
# Start the two self-hosted backends the integration tests run against: a
# single-tenant engine and a multitenant one (each test gets its own tenant).
set -euo pipefail

image="${RUSHDB_IMAGE:-rushdb/rushdb:3.0.4}"
token="${RUSHDB_TEST_ENGINE_TOKEN:-test-token}"

docker run -d --name rushdb-test -e RUSHDB_TOKEN="$token" -p 127.0.0.1:3917:3000 "$image"
# A multitenant engine refuses a pgwire listener without TLS; the tests use HTTP only.
docker run -d --name rushdb-test-mt -e RUSHDB_TOKEN="$token" -p 127.0.0.1:3918:3000 "$image" \
  --multitenant --pgwire-port 0

probe() {
  # probe <port> [scope]: the multitenant engine needs a scope, the other refuses one.
  local scope_header=() context=""
  if [ -n "${2:-}" ]; then
    scope_header=(-H "X-RushDB-Tenant: $2")
    context=",\"context\":{\"scope\":\"$2\"}"
  fi
  curl -sf -o /dev/null -X POST "http://127.0.0.1:$1/protocol/v1/execute" \
    -H "Authorization: Bearer $token" -H 'Content-Type: application/json' "${scope_header[@]}" \
    -d "{\"protocolVersion\":1$context,\"operation\":{\"kind\":\"describe\",\"describe\":{\"kind\":\"buildInfo\"}}}"
}

wait_ready() {
  # wait_ready <container> <port> [scope]
  for _ in $(seq 60); do
    probe "$2" "${3:-}" && return 0
    sleep 1
  done
  echo "$1 did not become ready" >&2
  docker logs "$1" >&2
  return 1
}

wait_ready rushdb-test 3917
wait_ready rushdb-test-mt 3918 probe
