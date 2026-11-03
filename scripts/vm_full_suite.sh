#!/usr/bin/env bash
# Run ON the TPU VM. Runs every single-chip TPU test target in the tree at a
# given revision, the way presubmit does (ci_tpu_v6 tag set minus multi-chip
# and RBE), and records one PASSED/FAILED/... line per target so two runs can
# be diffed. Usage: vm_full_suite.sh REV LABEL
#   REV    - commit, tag or origin/BRANCH to test
#   LABEL  - name for ~/fullsuite_LABEL.{log,results}
set -uo pipefail
REV=${1:?rev}; LABEL=${2:?label}
export PATH=$HOME/bin:$HOME/.local/bin:$PATH TMPDIR=/mnt/disks/build/tmp
cd ~/torch_tpu
LOG=~/fullsuite_$LABEL.log; RES=~/fullsuite_$LABEL.results
log() { echo "##### $* $(date -u +%H:%M:%S)" | tee -a "$LOG"; }
rm -f "$LOG" "$RES"
case "$REV" in origin/*) git fetch -q origin "${REV#origin/}" && REV=FETCH_HEAD;; esac
git checkout -q --detach "$REV" && log "checkout $LABEL: $(git log --oneline -1)"

# Single-chip TPU tests only: the ci_tpu_v6 positive set without the :4/:8
# multi-chip tags, v5p and v7x. 'manual' targets are excluded by //... itself.
TAGS="requires-tpu,requires-tpu-v6e,requires-tpu-v5lite,-requires-tpu-v5lite:4,-requires-tpu-v5lite:8,-requires-tpu-v5p,-requires-tpu-v5p:4,-requires-tpu-v7x,-requires-tpu-v7x:4,-notest,-nobuild,-fails-on-tpu-v6,-requires-newest-deps"

# //tests/... holds the Python op, error, compile and API suites; the C++
# unit tests under //csrc/ops (op_names_test and friends) run without the TPU
# tag filter since they are CPU tests. Everything else under //... (examples,
# experimental kernels) does not fit the 100 GB boot disk and is unrelated.
TARGETS="//tests/..."
log "build test targets"
bazel build --config=tpu_envvars --config=no_rbe --local_resources=cpu=HOST_CPUS*0.8 \
  --experimental_disk_cache_gc_max_size=8G \
  --build_tag_filters="$TAGS" -- $TARGETS > ~/fullsuite_${LABEL}_build.log 2>&1
echo "BUILD_EXIT=$?" | tee -a "$LOG"
df -h / | tail -1 | tee -a "$LOG"

log "csrc unit tests"
bazel test --config=no_rbe --keep_going --experimental_disk_cache_gc_max_size=8G \
  --cache_test_results=no --test_output=errors --test_summary=terse \
  -- //csrc/ops/... //csrc/common/... > ~/fullsuite_${LABEL}_csrc.log 2>&1
echo "CSRC_TEST_EXIT=$?" | tee -a "$LOG"

log "test"
bazel test --config=tpu_envvars --config=no_rbe --keep_going --local_test_jobs=1 \
  --experimental_disk_cache_gc_max_size=8G \
  --test_tag_filters="$TAGS" --test_timeout=600,1200,2400,4800 \
  --cache_test_results=no --test_output=errors --test_summary=terse \
  -- $TARGETS > ~/fullsuite_${LABEL}_test.log 2>&1
echo "TEST_EXIT=$?" | tee -a "$LOG"
cat ~/fullsuite_${LABEL}_csrc.log >> ~/fullsuite_${LABEL}_test.log

# One line per target: "//pkg:target STATUS". The terse summary lists every
# target with its status at the end of the log.
grep -E "^//[^ ]+ +(PASSED|FAILED|TIMEOUT|FLAKY|NO STATUS|\(cached\) PASSED)" ~/fullsuite_${LABEL}_test.log \
  | sed -E 's/ +/ /g; s/ in [0-9.]+s.*//; s/ \(cached\)//' | sort -u > "$RES"
echo "targets: $(wc -l < "$RES")  passed: $(grep -c ' PASSED' "$RES")  failed: $(grep -c ' FAILED' "$RES")  other: $(grep -vc ' PASSED\| FAILED' "$RES")" | tee -a "$LOG"
grep -v ' PASSED' "$RES" | tee -a "$LOG"
log "ALL DONE"
