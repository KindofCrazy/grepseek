import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=Path('mini/curated-qwen35-4b/sft_selected.jsonl'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--per-source',type=int,default=6)
    args=parser.parse_args()
    rows=[json.loads(line) for line in args.data.read_text(encoding='utf-8').splitlines()]
    selected=[]
    for source in sorted(set(row['source'] for row in rows)):
        candidates=sorted((row for row in rows if row['source']==source),
                          key=lambda row:hashlib.sha256(('42:'+row['id']).encode()).hexdigest())
        selected.extend(candidates[:args.per_source])
    cases=[{'trajectory_id':row['id'],'source':row['source'],'turn':turn,
            'recorded':json.loads(message['content'])}
           for row in selected for turn,message in enumerate(row['messages']) if message['role']=='tool']
    result={'sft_file_sha256':hashlib.sha256(args.data.read_bytes()).hexdigest(),
            'selection':'sha256(42:id), six per source by default, complete tool turns of each trajectory',
            'trajectory_ids':[row['id'] for row in selected],'cases':cases}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'trajectories':len(selected),'tool_calls':len(cases)}))


if __name__=='__main__': main()
