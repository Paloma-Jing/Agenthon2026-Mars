#!/usr/bin/env bash
# Build, push and pin the Track-2 submission image, then hand off to the packer.
#
# Steps, in order, each one checked before the next:
#   1. local structural smoke  (staged /input shape + official schema + g0-g3)
#   2. house-route rehearsal   (local stub validates bearer, /v1, token ceiling, call count)
#   3. build linux/amd64 for the evaluation fleet
#   4. hard checks on the built image: label present, no VOLUME, verb runs
#   5. push, then extract the immutable digest
#   6. write submission/image.json with that digest
#   7. print the anonymous-pull check and the packing command (packing is a separate step
#      because it asks for the Team Key)
#
# Usage:  submission/build_and_push.sh [--tag dev1] [--namespace YOUR_NAMESPACE] [--skip-smoke] [--no-push]
set -euo pipefail

REGISTRY="${REGISTRY:-docker.io}"
NAMESPACE="${NAMESPACE:-}"
REPO_NAME="${REPO_NAME:-qfb2-forecaster}"
TAG="dev1"
RUN_SMOKE=1
DO_PUSH=1
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

while [ $# -gt 0 ]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    --namespace) NAMESPACE="$2"; shift 2 ;;
    --skip-smoke) RUN_SMOKE=0; shift ;;
    --no-push) DO_PUSH=0; shift ;;
    -h|--help) sed -n '2,17p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

: "${NAMESPACE:?Set NAMESPACE or pass --namespace before building or publishing}"

IMAGE_REF="${REGISTRY}/${NAMESPACE}/${REPO_NAME}:${TAG}"

step() { printf '\n=== %s\n' "$*"; }

step "0/6 checking the toolchain"
PY="$ROOT/.venv-agent/bin/python"
[ -x "$PY" ] || { echo "missing $PY; run from the project root with .venv-agent present" >&2; exit 1; }
command -v docker >/dev/null 2>&1 || {
  echo "docker not found." >&2
  echo "Install Docker Desktop, start it once, then re-run this script." >&2
  exit 1
}
docker info >/dev/null 2>&1 || {
  echo "the docker daemon is not responding; start Docker Desktop and wait for it to finish starting" >&2
  exit 1
}
echo "docker $(docker version --format '{{.Server.Version}}')  python $("$PY" -V 2>&1)"

if [ "$RUN_SMOKE" = 1 ]; then
  step "1/6 structural smoke on a staged unit (no Docker needed)"
  rm -rf "$ROOT/runs/numeric_agent/staged_smoke"
  "$PY" "$ROOT/tools/smoke_staged_unit.py" --unit t2-F1-conflicting-texts-2024 \
        --out "$ROOT/runs/numeric_agent/staged_smoke" | tail -12

  step "2/6 house-route rehearsal against the local stub"
  rm -rf "$ROOT/runs/numeric_agent/house_stub"
  "$PY" "$ROOT/tools/smoke_house_stub.py" --out "$ROOT/runs/numeric_agent/house_stub" | tail -4
else
  step "1-2/6 smoke tests skipped on request"
fi

step "3/6 building $IMAGE_REF for linux/amd64"
cd "$ROOT"
docker buildx build --platform linux/amd64 -f Dockerfile.submission -t "$IMAGE_REF" --load .

step "4/6 contract checks on the built image"
LABEL_VALUE="$(docker image inspect "$IMAGE_REF" --format '{{index .Config.Labels "qfbench2.interface_version"}}')"
[ "$LABEL_VALUE" = "2.0" ] || { echo "label qfbench2.interface_version is '$LABEL_VALUE', expected '2.0'" >&2; exit 1; }
echo "label qfbench2.interface_version = $LABEL_VALUE"

VOLUMES="$(docker image inspect "$IMAGE_REF" --format '{{json .Config.Volumes}}')"
[ "$VOLUMES" = "null" ] || { echo "image declares VOLUME ($VOLUMES); a Final upload would be marked Failed" >&2; exit 1; }
echo "VOLUME declarations: none"

docker run --rm --platform linux/amd64 "$IMAGE_REF" forecast --help >/dev/null \
  || { echo "the forecast verb did not run inside the image" >&2; exit 1; }
echo "forecast verb: OK"

SIZE="$(docker image inspect "$IMAGE_REF" --format '{{.Size}}')"
echo "image size: $(( SIZE / 1024 / 1024 )) MiB (the cold pull comes out of each unit's wall clock)"

if [ "$DO_PUSH" = 0 ]; then
  step "5-6/6 --no-push: stopping after the local build"
  echo "re-run without --no-push to publish and pin a digest"
  exit 0
fi

step "5/6 pushing $IMAGE_REF"
docker push "$IMAGE_REF"

DIGEST="$(docker buildx imagetools inspect "$IMAGE_REF" | "$PY" -c 'import json,re,sys
text = sys.stdin.read()
m = re.search(r"Digest:\s*(sha256:[0-9a-f]{64})", text)
if m:
    print(m.group(1))
else:
    m = re.search(r"sha256:[0-9a-f]{64}", text)
    print(m.group(0) if m else "")' )"
if [ -z "$DIGEST" ]; then
  echo "could not read the digest automatically. Run this and read the Digest line:" >&2
  echo "  docker buildx imagetools inspect $IMAGE_REF" >&2
  exit 1
fi
echo "immutable digest: $DIGEST"

step "6/6 writing submission/image.json"
"$PY" - "$ROOT/submission/image.json" "$REGISTRY" "$NAMESPACE/$REPO_NAME" "$DIGEST" <<'PY'
import json, sys
path, registry, repository, digest = sys.argv[1:5]
with open(path, 'w') as handle:
    json.dump({'registry': registry, 'repository': repository, 'digest': digest},
              handle, indent=2)
    handle.write('\n')
print(open(path).read().strip())
PY

cat <<EOF

Next steps
----------
1. Anonymous pullability check (the evaluation worker has no registry credentials):

     docker logout
     docker pull ${REGISTRY}/${NAMESPACE}/${REPO_NAME}@${DIGEST}

2. Pack the upload. This asks for the Team Key on a hidden prompt and deletes its
   temporary file afterwards:

     .venv-agent/bin/python tools/prepare_upload.py --team-number YOUR_TEAM_NUMBER --phase dev

3. Upload submission/out/submission.zip to CodaBench unchanged.
EOF
