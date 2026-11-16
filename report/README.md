# report/

The technical report of the project: [`report_en.pdf`](report_en.pdf) (IEEEtran, two-column, English; status of
5 November 2026). It covers the system (G1, simulator, observation / action spaces), the method (PPO locomotion
policy, keyword target selection, Navigator, deployment), the results (walking emergence, closed-loop episodes), the
failure analysis, limitations and reproducibility. Earlier report in [`archive/`](archive/report.md).

**Status:** current (`report_en.pdf`); `archive/` kept for reference.

## Checking the numbers
Every number printed in the report is mapped to the repository file it was taken from (README, the archived
report, `media/README.md`, the code constants of the Navigator / episode protocol / evaluation) and checked by a
report-check script; an unmapped number or a stale mapping fails. The report also has to state that no success rate
is reported, that the Eagle embedding does not influence the target and that the system is not a
vision-language-action model. The numbers in the README, the archived report and `media/README.md` are recomputed
from the result files by the claims-check script. Both scripts, the LaTeX source and the figure script are part of
the code release, available on request (see the [README](../README.md#code-availability)).

## Layout
- `report_en.pdf`: the current report.
- `archive/report.{md,pdf}`: the Markdown engineering report (2026-11-01, was `docs/report.md`); the source of most of
  the current report's text and numbers, still checked by the claims check. Its repository paths are those of the
  2026-11 code layout.
