"""Check controlled-loop identity against frozen official inference at a paired seed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from hermit.io import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--atol',type=float,default=1e-3)
    parser.add_argument('--rtol',type=float,default=1e-3)
    args=parser.parse_args()
    import torch
    completed=json.loads((args.candidate/'complete.json').read_text())
    records=[]
    for identifier in completed['tasks']:
        reference=args.reference/identifier
        candidate=args.candidate/identifier
        old=json.loads((reference/'manifest.json').read_text())
        new=json.loads((candidate/'manifest.json').read_text())
        if old['task']!=new['task']:
            raise ValueError('parity comparison requires identical task inputs and seed')
        a=torch.load(reference/'latents.pt',map_location='cpu',weights_only=True).float()
        b=torch.load(candidate/'latents.pt',map_location='cpu',weights_only=True).float()
        if a.shape!=b.shape:
            raise ValueError('latent shapes differ')
        close=torch.allclose(a,b,atol=args.atol,rtol=args.rtol)
        records.append({'id':identifier,'passed':close,'max_abs':(a-b).abs().max().item(),
                        'mean_abs':(a-b).abs().mean().item(),'atol':args.atol,'rtol':args.rtol})
    result={'status':'passed' if all(r['passed'] for r in records) else 'failed','comparisons':records}
    atomic_json(args.candidate/'parity.json',result)
    print(json.dumps(result,indent=2))
    if result['status']!='passed': raise SystemExit(1)


if __name__=='__main__':
    main()
