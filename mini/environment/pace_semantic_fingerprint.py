"""Fingerprint the entire term/posting relation, independently of term IDs."""
import argparse
import json
from pathlib import Path
import subprocess


def fingerprint(index, binary):
    index=Path(index)
    metadata=json.loads((index/'meta.json').read_text())
    result=subprocess.run([str(binary),str(index),str(metadata['num_terms']),
        str(metadata['n_singletons']),str(metadata['num_lines'])],capture_output=True,text=True,check=True)
    return json.loads(result.stdout)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    source=Path(__file__).with_suffix('.c')
    binary=source.parents[1]/'.cache/pace-semantic-fingerprint'
    binary.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['cc','-O3','-Wall','-Wextra',str(source),'-ldl','-o',str(binary)],check=True)
    result=fingerprint(args.index,binary.resolve())
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
