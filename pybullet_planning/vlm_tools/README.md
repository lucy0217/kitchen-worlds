# Toolbox for querying vision-language models to guide task and motion planning

This is a guide to run example scripts for paper "Guiding Long-Horizon Task and Motion Planning with Vision Language Models" by Yang, et al. See [project page](https://zt-yang.github.io/vlm-tamp-robot/) or [arXiv](http://arxiv.org/abs/2410.02193).

## Paper experiment runner (GPT-4o-mini / local VLM)

Run `tutorials/run_vlm_paper.py` from the **kitchen-worlds repository root**.
This runner defaults to 2 sequencing methods × 3 reprompt budgets × 2 scene
difficulties × 2 arm settings × 30 trials = **720 TAMP trials**.
For each (method, difficulty, arm) condition it first generates **five independent
VLM responses** on the same seeded initial scene (40 candidate plans in total).
Responses are not forced to be distinct. Each trial samples one candidate uniformly
with replacement. The three reprompt budgets share the initial pool and paired
candidate selections / planning seeds, rather than generating 30 new initial plans.

```bash
conda activate kitchen
source your_project_folder/gpt_api.sh

# Five candidate plans + one TAMP trial: Easy, dual arm, subgoals, Nreprompt=1
python pybullet_planning/tutorials/run_vlm_paper.py \
  --output experiments/paper_smoke_new \
  --methods sequence --difficulties 0 --arms dual --reprompts 1 --trials 1

# Complete experiment: 40 candidates + 720 TAMP trials
python pybullet_planning/tutorials/run_vlm_paper.py \
  --output experiments/paper_gpt4o_mini
```

Use `--phase prepare` to generate only the candidate pools; `--phase evaluate`
requires all five candidates for every selected condition. The default `--phase all`
does both. Repeating the same command reuses completed jobs, including unsuccessful
TAMP tasks. A crashed job stops the batch and is not silently discarded or counted
as task failure; inspect its `worker.log` and use a new output directory for a rerun.
`--methods`, `--difficulties`, `--arms`, `--reprompts`, and `--trials` select subsets.
Each trial runs in a fresh subprocess; do not launch concurrent runners in the same
checkout because the original planner also has shared scratch directories.

For an already running local OpenAI-compatible vision server:

```bash
export LOCAL_VLM_URL=http://127.0.0.1:8000/v1/chat/completions
python pybullet_planning/tutorials/run_vlm_paper.py \
  --backend local --model Qwen/Qwen3-VL-8B-Instruct \
  --output experiments/paper_qwen8b
```

Use the model ID accepted by your server. Models must support image input. Model
selection never changes the prompt templates or TAMP settings. Local inference
must be configured separately and is not started by this script.

Protocol: temperature 0.2 for generation, translation and reprompting; diverse
planning enabled; `max_plans=max_evaluation_plans=12`; no learned task-plan
feasibility predictor; three object-reduction attempts per subproblem; action mode
retains the existing skeleton constraint checker. The runner calls `run_agent`
directly to avoid the legacy action-directory override to four skeletons.
Scene reconstruction uses the saved scene seed, while TAMP starts a separate
trial seed after initialization. Candidate sampling uses its own RNG.

The opt-in paper scene includes the fork and drawer omitted by the reduced demo,
opens the drawer in Easy, and randomizes initially open door angles slightly.
**The Easy lid remains on the stove**, as explicitly deferred; the hard scene has
the lid on the pot. Thus this is not an exact geometric reproduction of the paper.
Existing per-subproblem time limits remain 10s downward / 30s evaluation / 60s
stream and total planning; the paper excerpt does not specify those times.
The demo's extra eight-minute whole-run cutoff is disabled for this runner;
`--experiment-timeout SECONDS` can impose one explicitly. GUI defaults to off;
use `--viewer` when desired. The existing demo script retains its defaults.

Outputs: `protocol.json`, five `candidates/...` directories, and `trials/...`.
Every job saves parameters and seeds in `job.json`, the full console in `worker.log`,
and validated artifacts / metrics in `result.json`. `summary.csv` contains completed
trials, with repository metrics `task_success`, `continuous_success_rate`, and
`task_progress` preserved separately. A completed process does not imply task success.
No API keys are stored in these configuration files.

## Set up dependencies

First, follow the [main README in kitchen-worlds](https://github.com/Learning-and-Intelligent-Systems/kitchen-worlds) to install the dependencies.

Then run the following:

```shell
conda activate kitchen
pip install pddlgym anytree tabulate
```

## Set up pretrained VLM APIs

Set your OpenAI api key or Anthropic key in the environment variable (e.g. in `~/.bashrc` or `~/.zshrc`):

```shell
export OPENAI_API_KEY=<openai_api_key>
export ANTHROPIC_API_KEY=<anthropic_api_key>
```

or put them in a text file inside the `keys` directory that will be ignored by git

```shell
cd vlm_tools; mkdir keys
echo <openai_api_key> > keys/openai_api_key.txt
echo <anthropic_api_key> > keys/anthropic_api_key.txt
```

### Use an OpenAI-compatible API

The GPT client accepts an OpenAI-compatible base URL and defaults to
`gpt-4o-mini`. For Gate.AI, set the credentials in the shell that launches the
experiment:

```shell
export OPENAI_API_KEY='<your-new-api-key>'
export OPENAI_BASE_URL='https://api.gate.ai/openai/v1'
export OPENAI_MODEL='gpt-4o-mini'
export OPENAI_TIMEOUT=60
export OPENAI_MAX_RETRIES=3
export OPENAI_RETRY_BACKOFF=1
```

`OPENAI_MAX_RETRIES=3` allows three retries after the initial request for TLS,
connection, and timeout failures. `OPENAI_RETRY_BACKOFF=1` waits 1, 2, and 4
seconds between those retries. HTTP errors such as an invalid key or model are
reported immediately.

Do not commit the API key to this repository. If an API provider gives you a
full Chat Completions endpoint instead of a base URL, set it directly with
`OPENAI_CHAT_COMPLETIONS_URL`.

Select the GPT client when running the example because the default config uses
the local Qwen backend:

```shell
cd pybullet_planning
python tutorials/test_vlm_tamp.py \
    --vlm_type gpt4v \
    --open_goal "make chicken soup" \
    --exp_subdir "test_gate_gpt4o_mini" \
    --planning_mode "sequence"
```

### Run a local Transformers VLM

The default `config_nvidia_kitchen.yaml` uses a local OpenAI-compatible
endpoint backed by Qwen3-VL. The GPT and Claude API clients remain available
through `--vlm_type gpt4v` and `--vlm_type claude3`.

Start the local server in a separate terminal:

```shell
conda activate qwen-vl
hf download Qwen/Qwen3-VL-8B-Instruct \
    --local-dir ~/workspace/models/Qwen3-VL-8B-Instruct

cd ~/workspace/tapeout_2027/transformers
python local_vlm_server.py \
    --model-path ../../models/Qwen3-VL-8B-Instruct
```

The client defaults to `http://127.0.0.1:8000/v1/chat/completions`. These
environment variables can select another compatible endpoint or model:

```shell
export LOCAL_VLM_URL=http://127.0.0.1:8000/v1/chat/completions
export LOCAL_VLM_MODEL=Qwen/Qwen3-VL-8B-Instruct
export LOCAL_VLM_TIMEOUT=600
```

Use `--vlm_type local` or `--vlm_type qwen` to explicitly select the local
backend. A different Qwen checkpoint can be served without changing
`kitchen-worlds`; restart `local_vlm_server.py` with its local model path and
update `LOCAL_VLM_MODEL`.

To switch back to the 4B checkpoint, start the server with
`--model-path ../../models/Qwen3-VL-4B-Instruct` and set
`LOCAL_VLM_MODEL=Qwen/Qwen3-VL-4B-Instruct`.


## Run an example solving open-world kitchen problems

The planning modes `sequence` / `actions` allow no reprompting,
`sequence-reprompt1` / `actions-reprompt1` allow at most one reprompt,
and `sequence-reprompt` / `actions-reprompt` allow at most two reprompts.
For a single GPT run with `Nreprompt=1`, use
`python tutorials/test_vlm_tamp.py --vlm_type gpt4v --planning_mode sequence-reprompt1`.
Add `--dual_arm` to enable both arms. The configured limit is saved as
`max_reprompts` in `agent_memory.json`; `time.json` records the number actually used.

Ask GPT4v to break down a high-level goal to a sequence of subgoals, then give them to PDDLStream in sequence.

```shell
cd pybullet_planning
python tutorials/test_vlm_tamp.py \
    --open_goal "make chicken soup" \
    --exp_subdir "test_fun" \
    --planning_mode "sequence"
```

Common args to the script:
* `--open_goal`: Natural language description of the goal
* `--exp_subdir`: Output will be saved in `experiments/{exp_subdir}/{auto_datetime}_vlm-tamp/`
* `--problem_name`: Name to a python class in `vlm_tools/problems_vlm_tamp.py` that initiate the scene and problem, it initiates all objects that are required to solve the given open goal
* `--difficulty`: Difficulty level of the task, which the scene builder function uses to determine how much movable and articulated obstacles to add, `default=0`
* `--dual_arm`: `action='store_true'`, whether to use dual arm or single arm of the PR2 robot
* `--planning_mode`: Whether to use dual arm or single arm of the PR2 robot, `choices=['sequence', 'actions', 'sequence-reprompt', 'actions-reprompt']`
* `--load_llm_memory`: Subpath inside `kitchen-worlds/experiments/` in the format of `{exp_subdir}/{auto_datetime}_vlm-tamp/` where the previous responses generated by the VLM is saved, e.g. `test_run_vlm_tamp_pr2_chicken_soup/241106_212402_vlm-tamp`

The output logs of previous runs can be viewed at [http://0.0.0.0:9000/](http://0.0.0.0:9000/) by running the following in a different terminal.

```shell
(cd experiments/; python -m http.server 9000)
```

After the server is launched, the log of the last run can be viewed at [http://0.0.0.0:9000/latest_run/log/](http://0.0.0.0:9000/latest_run/log/).


## References 

Please cite the following paper if you use this code in your research:

```text 
@misc{yang2024guidinglonghorizontaskmotion,
      title={Guiding Long-Horizon Task and Motion Planning with Vision Language Models}, 
      author={Zhutian Yang and Caelan Garrett and Dieter Fox and Tomás Lozano-Pérez and Leslie Pack Kaelbling},
      year={2024},
      eprint={2410.02193},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2410.02193}, 
} 
```
