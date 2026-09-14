The template for a cluster in hpc should lool like this:
----------------
#!/bin/bash
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -N (sensible short name)
#$ -o logs/(sensible short name, possibly with subfolder)
#$ -q gpu.q
#$ -l gpu=1

source /usr/local/gpuallocation.sh
micromamba activate gpuenv
------------------

If you want to submit to cpu then you dont need to put gpuenv and gpu queue, you instead activate datasci environment (also in micromamba) and put it in the all.q.

Avoid set -euo pipefail. (It causes problems in the cluster)

Jobs should in general delete the logs from previous runs of the same job to avoid overwriting old job files and then we dont know which run caused each failure.

## Getting jobs onto CERES

There are two routes. Which one applies depends on where the agent is running.

### Route A: the agent runs on CERES (direct submission)

Check with `hostname` (`ceres.essex.ac.uk`) and `which qsub`. If both work, the agent is on the login node and can submit directly, **but only after the author approves that specific run**. Jobs consume shared cluster time, and approval for one run does not carry over to the next. Before asking, state the number of tasks, the queue, the concurrency limit (`-tc`) and a rough runtime estimate.

What worked in practice (KEEL benchmark, September 2026):
- Install the checkout into the job environment first (`pip install -e .` inside `datasci`). An old non-editable install makes jobs silently run stale code.
- Batch shells don't load the interactive profile. Activate explicitly with `eval "$(micromamba shell hook -s bash)"` followed by `micromamba activate datasci`.
- For many (dataset, method) pairs, use one array job (`qsub -t 1-N -tc 80`) that reads its line from a task list. Give each submission its own task-list file: a running array reads the file as tasks start, so overwriting a shared file repoints jobs that are still queued.
- Pin CPU jobs to one core (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`) and let the scheduler provide the parallelism.
- Write each result as its own file inside the repo, through a temporary file and a rename. Record failures as result files rather than letting them vanish, and skip pairs that already have a result, so resubmitting fills gaps.
- Run a small pilot array (a handful of pairs) before the full grid.
- Monitor with `qstat -u $USER` and the job logs. The scheduler's `usage` figures can lag; to check that a job is really working, run `ssh <node> ps -u $USER`.
- `all.q` has no runtime limit, so a single huge task can run for many hours. Estimate the cost of the largest dataset from a smaller one before submitting, and agree with the author what to do about outliers.
- Results are committed and pushed like any other change, once the author asks for it.

### Route B: the agent runs elsewhere (laptop, Burrita)

The agent cannot reach CERES. The loop is:

1. Agent writes job scripts + exact `qsub` commands into `COMMANDS.md`, and states the **exact result path/filename** each job will produce.
2. Author runs the jobs on CERES, then `git add` the result files, commit, and push.
3. Author pulls locally; the result files appear at the paths the agent expects.
4. Agent reads the committed result files as the sole interface (it does not "see" the cluster).

Consequences for how the agent must write jobs (these also hold for Route A):
- Every job's output path must be committed-and-pulled-friendly (relative to the repo, not scratch dirs on CERES).
- `COMMANDS.md` must pair each command with the file(s) to look for after pull, so the author knows exactly what to commit.
- The agent must not assume a run succeeded until the expected result file exists locally after a pull.
