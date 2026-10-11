"""Reproducible candidate pools and isolated trials for the VLM-TAMP paper.

Run from kitchen-worlds with the kitchen Python environment. See vlm_tools/README.md.
"""
import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def seed_for(seed, *parts):
    text = ':'.join(map(str, (seed,) + parts))
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) % (2**31)


def mode_for(method, reprompts):
    return method + {0: '', 1: '-reprompt1', 2: '-reprompt'}[reprompts]


def worker(job_path):
    job = json.loads(Path(job_path).read_text())
    for directory in ('', 'pybullet_planning', 'pddlstream', 'lisdf'):
        sys.path.insert(0, str(ROOT / directory))
    # Downstream parser must not see this runner's command line.
    sys.argv = [sys.argv[0]]
    from cogarch_tools.cogarch_run import run_agent
    from vlm_tools import modify_agent_args_for_vlm_tamp
    from vlm_tools.llamp_agent import LLAMPAgent, VLM_AGENT_CONFIG_ROOT
    from vlm_tools.problems_vlm_tamp import test_kitchen_chicken_soup

    if job['backend'] == 'gpt4v':
        os.environ['OPENAI_MODEL'] = job['model']
    else:
        os.environ['LOCAL_VLM_MODEL'] = job['model']

    def configure(args):
        args = modify_agent_args_for_vlm_tamp(args)
        # parse_agent_args normally enables GUI whenever a seed is supplied.
        args.viewer = job['viewer']
        return args

    work = Path(job['directory'])
    run_agent(
        problem=test_kitchen_chicken_soup, agent_class=LLAMPAgent,
        config='config_nvidia_kitchen.yaml', config_root=VLM_AGENT_CONFIG_ROOT,
        modify_agent_args_fn=configure, serve_page=False, use_learned_ir=True,
        world_builder_args={'difficulty': job['difficulty'], 'temp_dir': str(work / 'temp')},
        exp_dir=str(work), exp_subdir='runs', exp_name='vlm-tamp',
        api_class_name=job['backend'], open_goal='make chicken soup',
        llamp_planning_mode=job['mode'], dual_arm=job['arm'] == 'dual',
        seed=job['scene_seed'], planning_seed=job['planning_seed'],
        vlm_temperature=0.2, vlm_cache_dir=str(work / 'cache'),
        max_plans=12, max_evaluation_plans=12, use_heuristic_fc=False, data=True,
        object_reducer='object-related;heuristic-movables;all-joints',
        log_failures=job.get('log_failures', False),
        experiment_timeout=job['experiment_timeout'],
        save_testcase=job['phase'] == 'candidate',
        load_llm_memory=job.get('candidate_path'), viewer=job['viewer'],
    )
    runs = list((work / 'runs').glob('*/agent_memory.json'))
    if len(runs) != 1:
        raise RuntimeError('Expected exactly one run directory: {}'.format(work))
    run = runs[0].parent
    memory = json.loads(runs[0].read_text())
    key = 'lists_of_actions' if job['method'] == 'actions' else 'lists_of_subgoals'
    plans = memory.get(key + '_to_print', [])
    if len(plans) != 1 or not plans[0]:
        raise RuntimeError('Candidate must contain one nonempty parsed plan')
    result = {'status': 'completed', 'run_dir': str(run), 'initial_plan_length': len(plans[0])}
    if job['phase'] == 'trial':
        records = json.loads((run / 'time.json').read_text())
        summaries = [item for item in records if 'task_success' in item]
        if not summaries:
            raise RuntimeError('Run has no final planning summary')
        result['metrics'] = summaries[-1]
    write_json(work / 'result.json', result)


def execute(job):
    directory = Path(job['directory'])
    config_path = directory / 'job.json'
    result_path = directory / 'result.json'
    if config_path.exists():
        if json.loads(config_path.read_text()) != job:
            raise RuntimeError('Configuration differs from existing job: ' + str(config_path))
        if result_path.exists():
            return json.loads(result_path.read_text())
        raise RuntimeError('Incomplete job; inspect worker.log and use a new --output: ' + str(directory))
    write_json(config_path, job)
    print('Running ' + str(directory), flush=True)
    with (directory / 'worker.log').open('w') as log:
        env = os.environ.copy()
        env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
        env['MPLCONFIGDIR'] = str(directory / 'matplotlib')
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                    '--worker', str(config_path)], cwd=str(ROOT),
                                   env=env, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode or not result_path.exists():
        raise RuntimeError('Worker failed; see ' + str(directory / 'worker.log'))
    result = json.loads(result_path.read_text())
    print('Completed; task_success=' + str(result.get('metrics', {}).get('task_success', 'candidate')), flush=True)
    return result


def summarize(output):
    """Keep per-trial metrics (including unsuccessful tasks) easy to audit."""
    fields = ['method', 'difficulty', 'arm', 'mode', 'trial', 'candidate_id',
              'scene_seed', 'planning_seed', 'task_success', 'continuous_success_rate',
              'task_progress', 'total_plan_len', 'total_planning_time', 'run_dir']
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for result_path in sorted((output / 'trials').glob('*/*/*/result.json')):
            job = json.loads((result_path.parent / 'job.json').read_text())
            result = json.loads(result_path.read_text())
            writer.writerow(dict(job, **result['metrics'], run_dir=result['run_dir']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    parser.add_argument('--backend', choices=['gpt4v', 'local'], default='gpt4v')
    parser.add_argument('--model', default=None)
    parser.add_argument('--output', type=Path, default=ROOT / 'experiments/paper_gpt4o_mini')
    parser.add_argument('--phase', choices=['all', 'prepare', 'evaluate'], default='all')
    parser.add_argument('--methods', nargs='+', choices=['sequence', 'actions'], default=['sequence', 'actions'])
    parser.add_argument('--reprompts', nargs='+', type=int, choices=[0, 1, 2], default=[0, 1, 2])
    parser.add_argument('--difficulties', nargs='+', type=int, choices=[0, 1], default=[0, 1])
    parser.add_argument('--arms', nargs='+', choices=['single', 'dual'], default=['single', 'dual'])
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--seed', type=int, default=2027)
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--log-failures', action='store_true',
                        help='Record the concrete PDDLStream calls that fail while refining each skeleton')
    parser.add_argument('--experiment-timeout', type=int, default=0,
                        help='Optional whole-trial time cap in seconds; 0 disables the old 8-minute cap')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    if args.trials < 1 or args.experiment_timeout < 0:
        parser.error('trials must be positive and timeout nonnegative')
    search_path = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
    if not shutil.which('dot', path=search_path):
        parser.error('Graphviz dot is required in PATH or the current Python environment')
    args.output = args.output.resolve()
    model = args.model or ('gpt-4o-mini' if args.backend == 'gpt4v' else 'Qwen/Qwen3-VL-8B-Instruct')
    common = dict(backend=args.backend, model=model, viewer=args.viewer,
                  experiment_timeout=args.experiment_timeout, temperature=0.2,
                  max_plans=12, max_evaluation_plans=12, n_tamp=3,
                  log_failures=args.log_failures)
    spec = dict(common, seed=args.seed, candidates=5, easy_lid='existing stove placement')
    spec_path = args.output / 'protocol.json'
    if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
        raise RuntimeError('Output belongs to another protocol/model; choose a new --output')
    write_json(spec_path, spec)
    for method, difficulty, arm in itertools.product(args.methods, args.difficulties, args.arms):
        condition = '{}_d{}_{}'.format(method, difficulty, arm)
        scene_seed = seed_for(args.seed, 'scene', difficulty, arm)
        pool = []
        for candidate in range(5):
            directory = args.output / 'candidates' / condition / str(candidate)
            job = dict(common, phase='candidate', method=method, mode=method,
                       difficulty=difficulty, arm=arm, scene_seed=scene_seed,
                       planning_seed=None, candidate_id=candidate, directory=str(directory))
            if args.phase == 'evaluate':
                if not (directory / 'result.json').exists():
                    raise RuntimeError('Missing candidate; first run --phase prepare: ' + str(directory))
                if json.loads((directory / 'job.json').read_text()) != job:
                    raise RuntimeError('Candidate configuration mismatch: ' + str(directory))
                result = json.loads((directory / 'result.json').read_text())
            else:
                result = execute(job)
            pool.append(result['run_dir'])
        if args.phase == 'prepare':
            continue
        # Local RNG is never reset by simulation; paired selections across Nreprompt.
        for trial in range(args.trials):
            planning_seed = seed_for(args.seed, 'trial', difficulty, arm, trial)
            candidate = random.Random(seed_for(args.seed, condition, trial)).randrange(5)
            for reprompts in args.reprompts:
                directory = args.output / 'trials' / condition / ('r' + str(reprompts)) / ('trial_%03d' % trial)
                job = dict(common, phase='trial', method=method, mode=mode_for(method, reprompts),
                           difficulty=difficulty, arm=arm, scene_seed=scene_seed,
                           planning_seed=planning_seed, candidate_id=candidate,
                           candidate_path=pool[candidate], trial=trial, directory=str(directory))
                execute(job)
                summarize(args.output)
    print('Finished. Results: ' + str(args.output), flush=True)


if __name__ == '__main__':
    main()
