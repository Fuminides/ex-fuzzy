#!/bin/bash
# Freeze the EvoX GPU comparison grid and submit it to CERES as one gpu.q array job.
#
#   PILOT=1 bash benchmarks/cluster/submit_evox_gpu.sh    # 10,000 samples, 10 features, seed 7
#   bash benchmarks/cluster/submit_evox_gpu.sh            # the largest recorded CPU workloads
#   SAMPLES="10000 100000" MAX_CONCURRENT=8 bash benchmarks/cluster/submit_evox_gpu.sh
#   ROUTES="pymoo cpu device" bash benchmarks/cluster/submit_evox_gpu.sh   # add the PyMoo reference
#   FORCE=1 bash benchmarks/cluster/submit_evox_gpu.sh    # recompute finished workloads
#
# The default grid is the 100,000-sample row of docs/performance/t1_scaling.json:
# 10, 50 and 200 features, fixed and optimized partitions, seeds 7, 19 and 41,
# population 40 and 5 generations. Every task times the EvoX CPU routes and the
# EvoX GPU objective on the same GPU node.
#
# The jobs need EvoX importable in GPU_ENV (EvoX 1.3.0 works with PyTorch 2.6;
# 1.4.0 does not) and never import pymoo unless ROUTES includes pymoo.
#
# Results land in benchmarks/results/evox_gpu/n<samples>_d<features>_<partitions>_seed<seed>.json;
# a failed task leaves a .failed file with its log name instead.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/benchmarks/results/evox_gpu}"
LOG_DIR="${REPO_ROOT}/logs/evox_gpu"
GPU_ENV="${GPU_ENV:-gpuenv}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
GENERATIONS="${GENERATIONS:-5}"
POPULATION="${POPULATION:-40}"
ROUTES="${ROUTES:-cpu device}"
FORCE="${FORCE:-0}"
if [ "${PILOT:-0}" = "1" ]; then
  SAMPLES="${SAMPLES:-10000}"
  FEATURES="${FEATURES:-10}"
  SEEDS="${SEEDS:-7}"
fi
SAMPLES="${SAMPLES:-100000}"
FEATURES="${FEATURES:-10 50 200}"
PARTITIONS="${PARTITIONS:-fixed optimized}"
SEEDS="${SEEDS:-7 19 41}"

export MAMBA_EXE="${MAMBA_EXE:-$HOME/bin/micromamba}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba_env}"

if ! "${MAMBA_ROOT_PREFIX}/envs/${GPU_ENV}/bin/python" -c "import evox.operators" 2>/dev/null; then
  echo "EvoX cannot be imported in ${GPU_ENV}; install a compatible release first (e.g. evox==1.3.0)." >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
# Old logs would be indistinguishable from this run's; drop them first.
rm -f "${LOG_DIR}"/evoxgpu.o* "${LOG_DIR}"/evoxgpu.po*
# One frozen list per submission: an array job re-reads its task file as tasks
# start, so overwriting a shared file would repoint a run that is still going.
TASK_FILE="${OUTPUT_DIR}/tasks-$(date -u +%Y%m%dT%H%M%SZ).txt"
: > "${TASK_FILE}"
for samples in ${SAMPLES}; do
  for features in ${FEATURES}; do
    for partitions in ${PARTITIONS}; do
      for seed in ${SEEDS}; do
        echo "${samples} ${features} ${partitions} ${seed}" >> "${TASK_FILE}"
      done
    done
  done
done

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
  -v "REPO_ROOT=${REPO_ROOT},TASK_FILE=${TASK_FILE},OUTPUT_DIR=${OUTPUT_DIR},GPU_ENV=${GPU_ENV},GENERATIONS=${GENERATIONS},POPULATION=${POPULATION},ROUTES=${ROUTES},FORCE=${FORCE},MAMBA_EXE=${MAMBA_EXE},MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX}" \
  "${REPO_ROOT}/benchmarks/cluster/evox_gpu_array.qsub"
