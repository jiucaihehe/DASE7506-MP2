# Completed course submission

**Submitted successfully on 28 October 2026, 03:06:46 (UTC+8).**

- Course: DASE7506 MP2, Selecting Reliable LLM Reasoning.
- Student ID: **3036707472**.
- GitHub submitting account: **jiucaihehe**.
- Official full-test accuracy submitted: **83.94004282655246% (392/467)**.
- Actual submission issue: <https://github.com/xudongwu-0/xudongwu-0.github.io/issues/276>.
- Submitted source version: **590532406a947c8c9d7eaf775887a8a069ec64d4**.
- Code URL: <https://github.com/jiucaihehe/DASE7506-MP2/tree/590532406a947c8c9d7eaf775887a8a069ec64d4>.
- Complete checkpoint URL: <https://raw.githubusercontent.com/jiucaihehe/DASE7506-MP2/590532406a947c8c9d7eaf775887a8a069ec64d4/output/checkpoint/context_candidate.zip>.

## Completion checks

The course website generated the English issue title and body, including sealed
artifact URLs. Its generated JSON was preserved. The submission issue was actually
created using the student's existing native GitHub authentication after the GitHub
connector reported insufficient permission in the course repository. No credential
was stored in these records. The issue author's login, exact body, project,
student ID and full-precision score were checked after creation.

The course leaderboard workflow triggered by the issue completed successfully:
<https://github.com/xudongwu-0/xudongwu-0.github.io/actions/runs/40566138250>.
The authoritative leaderboard JSON, read through the GitHub contents API, was
generated at **2026-10-27T19:07:01.110Z** and contains MP2 submission **276** with
the correct account, student ID and score, and **artifact_status: received**.
The public code/checkpoint fields are null while encrypted links are sealed; this
is expected and is not evidence that the links were omitted. Public webpage/CDN
caches may refresh later than the underlying leaderboard data.

Both pinned artifact URLs were checked before submission. Unauthenticated downloads
of the instructor guide, evaluation helper, checkpoint ZIP, English PDF and official
score JSON returned HTTP 200 and matched local bytes. The ZIP contains exactly
`config.json` and `selected_bundle.npz`, matching the final inference directory.
The submitted model SHA-256 is unchanged:
`cc8483cb50c6b7287613897b5b78178b2accd0780888b5ac227d46f1fc984537`.

The source version includes all later development and frozen-evaluation disclosures.
The submitted entrypoint remains the 83.94% contextual model documented in
`../FINAL_SUBMISSION.md`. This submission step did not rerun test scoring, alter
the model, or replace the report.

## Evidence and limits

Saved records under `runs/course_submission/` include the website-generated issue
body, GitHub issue receipt, leaderboard receipt and a screenshot of the actual
created issue. The score remains **self-reported** in the course system; receipt
of the submission and artifact links does not mean the instructor has completed
checkpoint auditing or awarded a grade.

Any later changes to this repository do not change the immutable source/checkpoint
URLs submitted in issue 276. No further submission action is required to complete
this entry.
