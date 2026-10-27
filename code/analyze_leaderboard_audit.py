"""Summarize the public snapshot's displayed ties without inferring methods."""
import json
from collections import Counter
from pathlib import Path

out=Path(__file__).resolve().parent/'runs/submission_audit'
snapshot=json.loads((out/'leaderboard.json').read_text(encoding='utf-8'))
best={}
for row in snapshot['submissions']:
    if row.get('project')!='mp2' or row.get('status') not in ('self-reported','verified'): continue
    key=(row['student_id'],row['author'].lower())
    old=best.get(key)
    if old is None or row['score']>old['score']: best[key]=row
rows=sorted(best.values(),key=lambda r:-r['score'])
counts=Counter(format(r['score'],'.4f') for r in rows)
result=dict(snapshot_generated_at=snapshot['generated_at'],best_rows=len(rows),
    displayed_ties={s:n for s,n in counts.items() if n>1},
    status_counts=dict(Counter(r['status'] for r in rows)),
    score_step_percentage_points=100/467,
    illustrative_correct_counts={str(n):100*n/467 for n in (390,391,392)},
    own_account_records=[dict(issue=r['number'],score=r['score'],artifact_status=r['artifact_status'])
                         for r in rows if r['author'].lower()=='jiucaihehe'])
(out/'leaderboard_analysis.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
