#!/usr/bin/env bash
# DreamDrive release QA orchestrator (make qa). Runs every gate in order against the committed source, stores a
# log, exit code, duration and evidence directory per gate in artifacts/qa/<commit>/, and writes summary.json +
# summary.md. The run FAILS if any gate fails, is blocked by a failed prerequisite, is not run, or reports
# skipped tests (pytest/Playwright junit "skipped" > 0): skips never count as passes.
#
#   make qa                       # clean tree required (uncommitted changes are not release evidence)
#   QA_ALLOW_DIRTY=1 make qa      # allowed for iteration; summary is marked "release_evidence": false
#   QA_ONLY="lint unit" make qa   # subset for debugging; every other gate is recorded as not_run => FAIL
#   QA_KEEP_UP=1 make qa          # leave the stack running afterwards (default: make down, volumes kept)
set -Euo pipefail
DD_LOG_TAG=qa
# shellcheck source=deploy/lib/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/../deploy/lib/common.sh"
cd "$DD_REPO_ROOT" || exit 1

commit=$(git rev-parse HEAD 2>/dev/null) || die "make qa needs a git commit" 2
dirty=0
[[ -z $(git status --porcelain) ]] || dirty=1
if ((dirty)) && [[ ${QA_ALLOW_DIRTY:-0} != 1 ]]; then
  git status --short | head -n 20 >&2
  die "working tree has uncommitted changes; commit first (or QA_ALLOW_DIRTY=1 for a non-release run)" 2
fi
run_id=$commit
((dirty)) && run_id="$commit-dirty-$(utc_stamp)"
qa_dir="artifacts/qa/$run_id"
mkdir -p "$qa_dir"
export DD_REQUIRE_LIVE=1
fixture_clip="tests/fixtures/cache/commons_dashcam_daylight_cc0_96s_10s.mp4"

# name | prerequisites | command (run from the repo root; EVIDENCE_DIR is appended for make targets)
gates=(
  "doctor||make doctor"
  "deploy-check||make deploy-check"
  "ops-check||make ops-check"
  "ops-selftest||make deploy-selftest mlflow-selftest"
  "build||make build build-test"
  "lint|build|make lint"
  "typecheck|build|make typecheck"
  "gen-types|build|make gen-types-check"
  "unit|build|make test"
  "fixtures|build|make fixtures"
  "model-init|build|make model-init"
  "model-smoke|model-init fixtures|make model-smoke"
  "stack-up|model-init|make up"
  "integration|stack-up fixtures|make integration"
  "scenarios|stack-up|make scenarios"
  "smoke-upload|stack-up fixtures|make smoke UPLOAD=$fixture_clip"
  "e2e|stack-up fixtures|make e2e"
  "eval-scenarios|stack-up|make eval-scenarios"
  "security|stack-up|make security"
  "backup-test|integration smoke-upload|make backup-test"
  "sbom|build|make sbom"
)

{
  echo "date_utc: $(utc_now)"
  echo "git_commit: $commit"
  echo "dirty: $dirty"
  echo "uname: $(uname -a)"
  echo "docker: $(docker version --format 'client {{.Client.Version}} / engine {{.Server.Version}} {{.Server.Os}}/{{.Server.Arch}}' 2>&1)"
  echo "compose: $(docker compose version --short 2>&1)"
  echo "docker_vm: $(docker info --format '{{.NCPU}} CPU, {{.MemTotal}} bytes RAM' 2>&1)"
  echo "make: $(make --version 2>&1 | head -n 1)"
} >"$qa_dir/environment.txt"

declare -A status
: >"$qa_dir/gates.tsv"
printf 'n\tgate\tstatus\texit_code\tseconds\tskipped_tests\tlog\n' >>"$qa_dir/gates.tsv"

junit_skipped() { # sum of skipped="N" attributes in junit files below DIR (0 if none)
  local dir=$1
  [[ -d $dir ]] || {
    echo 0
    return
  }
  find "$dir" -name '*.xml' -type f -print0 | xargs -0 -r grep -ho '<testsuite[^>]* skipped="[0-9]*"' 2>/dev/null |
    sed -E 's/.* skipped="([0-9]+)"/\1/' | awk '{s += $1} END {print s + 0}'
}

n=0
for spec in "${gates[@]}"; do
  IFS='|' read -r name deps cmd <<<"$spec"
  n=$((n + 1))
  num=$(printf '%02d' "$n")
  log_file="$qa_dir/$num-$name.log"
  evidence="./$qa_dir/$name"
  if [[ -n ${QA_ONLY:-} && " $QA_ONLY " != *" $name "* ]]; then
    status[$name]=not_run
    printf '%s\t%s\tnot_run\t-\t0\t-\t-\n' "$num" "$name" >>"$qa_dir/gates.tsv"
    log "[$num] $name: not_run (QA_ONLY)"
    continue
  fi
  blocked=""
  for dep in $deps; do
    [[ ${status[$dep]:-} == pass ]] || blocked+="$dep "
  done
  if [[ -n $blocked ]]; then
    status[$name]=blocked
    printf '%s\t%s\tblocked\t-\t0\t-\tprerequisite not passed: %s\n' "$num" "$name" "$blocked" >>"$qa_dir/gates.tsv"
    log "[$num] $name: BLOCKED (prerequisite not passed: $blocked)"
    continue
  fi
  mkdir -p "$evidence"
  # Linux: lets the container uid 10001 write junit files into the bind mount (no effect on Windows mounts).
  if ! chmod 0777 "$evidence" 2>/dev/null; then warn "could not chmod $evidence"; fi
  log "[$num] $name: $cmd"
  started=$SECONDS
  rc=0
  # shellcheck disable=SC2086 # cmd is a trusted, space-separated make invocation defined above
  $cmd EVIDENCE_DIR="$evidence" >"$log_file" 2>&1 || rc=$?
  seconds=$((SECONDS - started))
  skipped=$(junit_skipped "$evidence")
  if ((rc != 0)); then
    status[$name]=fail
  elif ((skipped > 0)); then
    status[$name]=fail_skipped
  else
    status[$name]=pass
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$num" "$name" "${status[$name]}" "$rc" "$seconds" "$skipped" \
    "$(basename "$log_file")" >>"$qa_dir/gates.tsv"
  log "[$num] $name: ${status[$name]} (exit $rc, ${seconds}s, skipped tests: $skipped)"
  if [[ ${status[$name]} != pass ]]; then tail -n 15 "$log_file" | sed 's/^/    | /' >&2; fi
done

# Stack evidence before shutdown (service states and logs; application logs never contain secrets by contract).
if docker compose -f compose.yaml ps -q 2>/dev/null | grep -q .; then
  docker compose -f compose.yaml ps >"$qa_dir/stack-ps.txt" 2>&1 || warn "could not record compose ps"
  docker compose -f compose.yaml logs --no-color --timestamps >"$qa_dir/stack-logs.txt" 2>&1 ||
    warn "could not record stack logs"
fi
{
  for image in dreamdrive-web dreamdrive-backend dreamdrive-inference dreamdrive-backend-test dreamdrive-inference-test; do
    printf '%s:%s %s\n' "$image" "${DD_TAG:-dev}" \
      "$(docker image inspect --format '{{.Id}}' "$image:${DD_TAG:-dev}" 2>/dev/null || echo missing)"
  done
} >"$qa_dir/local-image-ids.txt"
if [[ ${QA_KEEP_UP:-0} != 1 ]]; then
  make down >"$qa_dir/zz-down.log" 2>&1 || warn "make down failed (see $qa_dir/zz-down.log)"
fi

passed=0 total=0 overall=pass
for spec in "${gates[@]}"; do
  name=${spec%%|*}
  total=$((total + 1))
  if [[ ${status[$name]} == pass ]]; then passed=$((passed + 1)); else overall=fail; fi
done
model_sha=$(sed -n 's/^ *"checkpoint_sha256": "\([0-9a-f]\{64\}\)".*/\1/p' config/models.resolved.yaml | head -n 1)
release_evidence=true
((dirty)) && release_evidence=false
{
  printf '{\n  "schema": "dreamdrive-qa/1",\n  "git_commit": "%s",\n  "dirty": %s,\n' "$commit" "$([[ $dirty == 1 ]] && echo true || echo false)"
  printf '  "release_evidence": %s,\n  "finished_utc": "%s",\n  "result": "%s",\n' "$release_evidence" "$(utc_now)" "$overall"
  printf '  "passed": %d,\n  "total": %d,\n  "model_sha256": "%s",\n  "gates": {\n' "$passed" "$total" "$model_sha"
  i=0
  for spec in "${gates[@]}"; do
    name=${spec%%|*}
    i=$((i + 1))
    sep=","
    ((i == total)) && sep=""
    printf '    "%s": "%s"%s\n' "$name" "${status[$name]}" "$sep"
  done
  printf '  }\n}\n'
} >"$qa_dir/summary.json"
{
  echo "# DreamDrive QA $run_id"
  echo
  echo "Result: **${overall^^}** ($passed/$total gates passed). Release evidence: $release_evidence."
  echo
  echo "| # | Gate | Status | Exit | Seconds | Skipped tests |"
  echo "|---|---|---|---|---|---|"
  tail -n +2 "$qa_dir/gates.tsv" | awk -F'\t' '{printf "| %s | %s | %s | %s | %s | %s |\n", $1, $2, $3, $4, $5, $6}'
  echo
  echo "Model checkpoint SHA256: \`$model_sha\`. Environment: environment.txt. Logs: NN-<gate>.log."
} >"$qa_dir/summary.md"

cat "$qa_dir/summary.md"
log "QA $overall: $passed/$total gates passed; evidence in $qa_dir"
[[ $overall == pass ]]
