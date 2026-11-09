"""Pure-stdlib worker, executed inside the credential-free search sandbox."""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--command',required=True)
    parser.add_argument('--shards',required=True)
    parser.add_argument('--corpus',required=True)
    parser.add_argument('--timeout',type=float,required=True)
    args = parser.parse_args()
    sys.path.insert(0,'/app')
    from inference.parallel_search.engine import ShardedSearchEngine
    engine = ShardedSearchEngine(shard_dir=args.shards,fallback_corpus_file=args.corpus,timeout=args.timeout)
    stdout,code = engine.execute(args.command,env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8'})
    sys.stdout.write(stdout)
    if code not in (0,1):
        sys.stderr.write('sharded_executor_failed\n')
    return code


if __name__ == '__main__':
    sys.exit(main())
