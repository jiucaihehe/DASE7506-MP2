"""Select from validation only, then freeze a self-contained inference asset folder.

This script never loads test inputs or references. Run before test evaluation.
"""
import hashlib
import json
import shutil
from pathlib import Path

OUT=Path('runs/final_candidate')


def main():
    candidates=[dict(name='original',family='original',hard_correct=367,all_correct=746,
                     checkpoint='runs/model_bundle.npz')]
    experiments=[('bilinear','bilinear_experiment'),('linear','objective_experiment'),
                 ('kernel','kernel_experiment'),('structural','structural_experiment'),('bagging','bagging_experiment')]
    for family,folder in experiments:
        path=Path('runs')/folder
        r=json.loads((path/'validation_results.json').read_text())
        candidates.append(dict(name=folder,family=family,hard_correct=r['selected']['hard_correct'],
            all_correct=r['selected']['all_correct'],checkpoint=str(path/'selected_bundle.npz')))
    # Finite method selection on validation is permitted. No score from test enters here.
    winner=max(candidates,key=lambda c:(c['hard_correct'],-Path(c['checkpoint']).stat().st_size))
    if OUT.exists(): raise FileExistsError('Refusing to overwrite an already frozen candidate')
    OUT.mkdir(parents=True)
    shutil.copyfile(winner['checkpoint'],OUT/'selected_bundle.npz')
    sha=hashlib.sha256((OUT/'selected_bundle.npz').read_bytes()).hexdigest()
    config=dict(family=winner['family'],checkpoint_sha256=sha,
        selected_using='validation hard accuracy; asset size breaks ties',
        validation_hard_correct=winner['hard_correct'],validation_hard_n=419,
        validation_correct=winner['all_correct'],validation_n=798,
        train_only_fit=True,test_used_for_selection=False)
    (OUT/'config.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    freeze=dict(candidates=candidates,selected=winner,config=config,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in map(Path,
            ['submission_candidate.py','method_bilinear.py','method_bagging.py','method_kernel.py',
             'structural_text.py','program_transfer.py','features.py','common.py'])},
        inference_asset_bytes=sum(p.stat().st_size for p in OUT.iterdir()))
    freeze['source_sha256_lf']={p:hashlib.sha256(Path(p).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                                for p in freeze['source_sha256']}
    (Path('runs')/'candidate_freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
    print(json.dumps(freeze,indent=2))


if __name__=='__main__': main()
