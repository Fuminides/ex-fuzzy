#!/bin/bash
# Freeze the (dataset, method) grid and submit it to CERES as one array job.
#
#   bash benchmarks/cluster/submit_keel.sh              # whole collection
#   MAX_CONCURRENT=60 bash benchmarks/cluster/submit_keel.sh
#   DATASETS="iris wine" bash benchmarks/cluster/submit_keel.sh
#   FORCE=1 bash benchmarks/cluster/submit_keel.sh    # recompute finished pairs
#
# Resubmitting without FORCE=1 skips pairs that already have a result file, so
# it is the way to fill gaps after a partial run.
#
# Results land in benchmarks/results/keel/<dataset>__<method>.json, which is
# what benchmarks/aggregate_keel.py reads.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/benchmarks/results/keel}"
# One frozen list per submission: an array job re-reads its task file as tasks
# start, so overwriting a shared file would repoint a run that is still going.
TASK_FILE="${OUTPUT_DIR}/tasks-$(date -u +%Y%m%dT%H%M%SZ).txt"
LOG_DIR="${REPO_ROOT}/logs/keel"
KEEL_ENV="${KEEL_ENV:-datasci}"
MAX_CONCURRENT="${MAX_CONCURRENT:-80}"
FOLDS="${FOLDS:-5}"
SEED="${SEED:-0}"
FORCE="${FORCE:-0}"

export MAMBA_EXE="${MAMBA_EXE:-$HOME/bin/micromamba}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba_env}"
eval "$("${MAMBA_EXE}" shell hook --shell bash --root-prefix "${MAMBA_ROOT_PREFIX}")"
micromamba activate "${KEEL_ENV}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
# Old logs would be indistinguishable from this run's; drop them first.
rm -f "${LOG_DIR}"/keelbench.o* "${LOG_DIR}"/keelbench.po*

cd "${REPO_ROOT}" || exit 2
LIST_ARGS=(--list-tasks --order size)
if [ -n "${DATASETS}" ]; then LIST_ARGS+=(--datasets ${DATASETS}); fi
if [ -n "${METHODS}" ]; then LIST_ARGS+=(--methods ${METHODS}); fi
python benchmarks/benchmark_keel.py "${LIST_ARGS[@]}" > "${TASK_FILE}"

TASKS=$(grep -c . "${TASK_FILE}")
if [ "${TASKS}" -lt 1 ]; then
  echo "Empty task list at ${TASK_FILE}" >&2
  exit 2
fi
echo "Submitting ${TASKS} tasks (max ${MAX_CONCURRENT} concurrent) from ${TASK_FILE}"

qsub \
  -t "1-${TASKS}" \
  -tc "${MAX_CONCURRENT}" \
  -o "${LOG_DIR}" \
  -v "REPO_ROOT=${REPO_ROOT},TASK_FILE=${TASK_FILE},OUTPUT_DIR=${OUTPUT_DIR},KEEL_ENV=${KEEL_ENV},FOLDS=${FOLDS},SEED=${SEED},FORCE=${FORCE},MAMBA_EXE=${MAMBA_EXE},MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX}" \
  "${REPO_ROOT}/benchmarks/cluster/keel_array.qsub"
