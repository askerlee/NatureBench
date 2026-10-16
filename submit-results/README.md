# Submit Results to the NatureBench Leaderboard

We welcome submissions from researchers and developers who have evaluated new
models or agents on NatureBench. All tracks use the same submission format,
validation tools, and review process.

## Choose a track

| Track | Track ID | Tasks |
|---|---|---:|
| Full | `full` | 90 |
| NatureBench-25 | `naturebench-25` | 25 |
| Cellular Omics | `cellular-omics` | 31 |
| Protein Biology | `protein-biology` | 16 |
| Biomedical Modeling | `biomedical-modeling` | 14 |
| Physical Modeling | `physical-modeling` | 13 |
| Molecular Design | `molecular-design` | 11 |
| Relational Reasoning | `relational-reasoning` | 5 |

Prepare one package per track, covering every official task in that track,
including failed, invalid, and no-score runs. To submit multiple tracks, prepare
a separate directory for each. The six domain tracks use the Full benchmark's
domain task sets.

## Prepare the submission

A submission directory contains:

```text
<submission-name>/
├── submission.yaml
├── results.csv
└── raw-results/
    ├── <case_id>/
    └── README.md              # when format or review notes are needed
```

From the repository root, generate the metadata template, result rows, and task list:

```bash
python submit-results/prepare_submission.py \
  --track full \
  --output-dir my-submission
```

Replace `full` with any track ID above. The command creates `submission.yaml`,
`results.csv`, and `tasks.txt` for the selected track.
Complete the `submission.yaml` and `results.csv`, and add the per-case `raw-results/`.

You can also start from [`templates/submission.yaml`](templates/submission.yaml)
and the existing [CSV templates](templates/).
See [`SUBMISSION_SPEC.md`](SUBMISSION_SPEC.md) for instructions on completing `submission.yaml` and `results.csv`, organizing the
per-case raw results, and the result review and publication process.

### Continuing a NatureBench-25 batch

If two tasks are finished and the other 23 need fresh runs, keep the full
`task-set/naturebench-25/all.txt` selection and the same output directory:

```bash
python run_naturebench.py ... \
  --tasks ./task-set/naturebench-25/all.txt \
  --out-dir ./results/my_run \
  --force-fresh-task-file ./fresh_23.txt

python submit-results/prepare_submission.py \
  --track naturebench-25 \
  --output-dir my-submission
```

List only the 23 tasks to restart in `fresh_23.txt`; finished tasks outside that
list are automatically skipped. Use a separate, non-overlapping resume list
for tasks that should continue instead. Do not use `--resume-only` or replace
the full selection with the 23-task list: either intentionally narrows coverage.

The preparation command creates 25 CSV rows, not completed results. Manually
complete metadata and every row accurately using the cumulative outputs and
per-case evidence, including the two old tasks and all newly run tasks:

- Copy `final_status` from each task's `result.json.status`; report the best
  evaluator aggregate improvement as `best_score`, leaving it empty if unscored.
  Timeout alone implies neither success nor a score.
- Numeric scores require `valid` or `invalid` and the exact saved judge reason.
  For unscored tasks without a judge verdict, use `not_applicable` for both
  judge fields. Do not invent missing scores or verdicts.
- Copy per-case raw artifacts from both old and new tasks into
  `raw-results/<case_id>/`: `result.json`, `submissions.jsonl` when scored,
  `judge_verdict.json` for valid/invalid verdicts, and readable, non-empty
  `trajectory.*` files covering the complete run history (copy execution logs
  into that naming convention when needed).
- Remove credentials, session authentication files, and tokens, including any
  embedded in logs or optional workspaces. Exclude archived task directories
  and unrelated directories; copy the required evidence, not the whole outdir.

The cumulative `run_summary.json` helps reconcile all 25 tasks but is not a
submission package. Neither running nor preparing results submits anything
automatically.

## Check the submission

From the repository root, run:

```bash
python submit-results/validate_submission.py \
  --metadata <submission-name>/submission.yaml \
  --results <submission-name>/results.csv \
  --raw-results <submission-name>/raw-results

python submit-results/compute_scores.py \
  --track <track-id> \
  --results <submission-name>/results.csv \
  --output <submission-name>/score_report.json
```

Use the same track ID as `evaluation.track` in `submission.yaml`.
The first command checks the submission format and the consistency of the
available result records according to `evaluation_pipeline`. For `custom`
pipelines, it only checks case coverage and non-empty raw artifacts.
The second command computes a preview of the public metrics and optionally
writes a standalone JSON report.

## Send the submission

Remove credentials and private endpoints, archive the directory, and send it
or a stable download link to **wangyuru@frontis.cn, zhangkaiyan@frontis.cn** with the subject:

```text
[NatureBench Result Submission] <agent> + <model>
```

Please include a short introduction and links to the model, agent, paper, or
project when available. Maintainers will review the artifacts, rerun the
validity judge when needed, and contact the submitter if more
information is required. Submissions that pass review are scored with
[`compute_scores.py`](compute_scores.py) and then added to the leaderboard.
