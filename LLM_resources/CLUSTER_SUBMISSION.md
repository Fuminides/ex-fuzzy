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

## Result round-trip protocol (agent <-> CERES)

The agent cannot submit to CERES. The loop is:

1. Agent writes job scripts + exact `qsub` commands into `COMMANDS.md`, and states the **exact result path/filename** each job will produce.
2. Author runs the jobs on CERES, then `git add` the result files, commit, and push.
3. Author pulls locally; the result files appear at the paths the agent expects.
4. Agent reads the committed result files as the sole interface (it does not "see" the cluster).

Consequences for how the agent must write jobs:
- Every job's output path must be committed-and-pulled-friendly (relative to the repo, not scratch dirs on CERES).
- `COMMANDS.md` must pair each command with the file(s) to look for after pull, so the author knows exactly what to commit.
- The agent must not assume a run succeeded until the expected result file exists locally after a pull.