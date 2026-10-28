"""Score all completed cases in a run, retaining failures and per-metric results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from examples.video_transition.utils.evaluate_video import evaluate
from examples.video_transition.utils.remote import load_settings


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args()
    settings=load_settings(args.runtime)
    run=args.run_dir.resolve()
    complete=json.loads((run/'complete.json').read_text())
    results,errors=[],[]
    for identifier in complete['tasks']:
        try:
            case=run/identifier
            manifest=json.loads((case/'manifest.json').read_text())
            if manifest['status']!='complete': raise ValueError('case execution was not successful')
            results.append(evaluate(manifest['task'],Path(manifest['video_path']),settings,case/'evaluation'))
        except Exception as error:
            errors.append({'id':identifier,'error':str(error)})
    summary={'status':'partial' if errors else 'complete','expected_cases':len(complete['tasks']),
             'completed_cases':len(results),'errors':errors,
             'cases':[{key:value for key,value in row.items() if key!='feedback'} for row in results]}
    if not errors:
        names={name for row in results for name in row['metrics']}
        summary['metrics']={name:sum(row['metrics'][name] for row in results if name in row['metrics']) /
                            sum(name in row['metrics'] for row in results) for name in sorted(names)}
        summary['score']=sum(summary['metrics'].values())/len(summary['metrics'])
    (run/'evaluation-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({key:value for key,value in summary.items() if key!='cases'},indent=2))
    if errors: raise SystemExit(1)


if __name__=='__main__':
    main()
