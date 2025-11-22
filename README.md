# CPR Simulation with LLM Agents

This repository accompanies the blog here:
[https://loui.uk/AISafety](https://loui.uk/AISafety)


---

## Requirements

To reproduce the runs, fill in the `template.secrets` file with your API keys, then rename it to
`.secrets`.
To export the environment variables in the current session you can then use
```bash
export $(grep -v '^#' .secrets | xargs)
```

The python environment must have these packages:
```bash
pip install pillow anthropic google-api-python-client litellm matplotlib numpy openai pandas scipy seaborn tqdm
```
I made a requirements file if you prefer.

If you want a compressed file containing the original runs you must run:
```bash
git lfs pull
```

---

## Hardcoded Runs

Firstly there is `hardcoded_runs.py` this is useful for calibrating the message costs /
messaging amounts / resource dynamics of the simulations with simple hardcoded agents.
```bash
python hardcoded_runs.py
```
These hard-coded runs matches up with the "Common Pool Resource Simulation" section of the blog.

---

## Common Pool Resource Simulation with Agents

I have kept `simulate_cpr.py` which has a subset of the functionality of `snitch_cpr.py`
because it might be useful for users to see a slimmed down (but still massive) script to branch off
or just understand the main common pool resource simulation system. I've put everything
in one script which I will probably get hate for but it's just a research script not a
library, I would like to refactor snitch CPR at some point- to organise the agents feeds
very differently, with an append to next prompt buffer as it would clean up the flow of
code. So before I extend the simulation I will do this.
```bash
# To start off a new run which will be written to the runs directory
python simulate_cpr.py \
    --seed 0 \
    --directory runs \
    --agent-models openai/gpt-5-mini openai/gpt-5-mini openai/gpt-5-mini openai/gpt-5-mini

# To resume a checkpoint from the last completed round, useful if a API call failed
python simulate_cpr.py \
    --seed 0 \
    --directory runs \
    --run-index 35

# To start off a new run prompting Member1 with the saboteur prompt
python simulate_cpr.py \
    --seed 0 \
    --directory runs \
    --agent-models openai/gpt-5-mini openai/gpt-5-mini openai/gpt-5-mini openai/gpt-5-mini \
    --saboteur Member1
```
If you want to run many experiments at once then `launch_experiments.py` might be
helpful, be sure to edit the script to make sure it is running the experiments you want.
```bash
python launch_experiments.py
```
It simply starts multiple experiments.
You can view all of the outputs in one terminal window with:
```bash
python listener.py
```
it makes it easy to see the status of runs, e.g. if they have failed due to rate limits.

When doing the runs you should update a `completed_runs.json` file which has the
experiment group name and then the experiment names with a list of all the run indices
for each seed. E.g.
```json
{
    "regular": {
        "GPT-5 Nano": [34, 45, 54, 62, 73],
        "Gemini-2.5 Flash": [...],
        "DeepSeek Reasoner": [],
        "Claude Haiku 4.5": [],
        "GPT-5 Mini": [],
        "GPT-5": [],
        "Kimi K2 Thinking": [],
        "Mixed Companies": [],
        "Mixed Intelligence": []
    },
    "saboteur": {
        "GPT-5 Saboteur": [],
        "Kimi K2 Thinking Saboteur": [],
        "DeepSeek Reasoner Saboteur": [],
        "Claude Haiku 4.5 Saboteur": []
    },
    "saboteur_with_snitches": {
        "Kimi Saboteur with GPT-5 Mini Snitches": []
    },
    "bad_kimi_rollouts": {
        "From start, no snitches": [],
        "From R2, no snitches": [],
        "From R4, no snitches": [],
        "From R6, no snitches": [],
        "From start, snitches": [],
        "From R2, snitches": [],
        "From R4, snitches": [],
        "From R6, snitches": []
    },
}
``` 

---

## Analyse the Runs

You can render the dashboard animations with:
```bash
python render_dashboard.py --run-path runs/run_1/main_log.json
```
for the run with index 1.
If you just want an image of the final plots as an image use:
```bash
python render_dashboard.py --run-path runs/run_1/main_log.json --last-only
```
If you want the conversation animation in the card on the blog use:
```bash
python render_conversation.py
```
but you must edit the script to define the run directory and the member to follow.

You can also generate a html file for run 1 which makes it much easier to read through
the conversations.
```bash
python generate_conversation_html.py --run-path runs/run_1
```


The `completed_runs.json` file is needed to do the comparison analysis.


To produce dashboard plots for all experiments run `quantitative_plots.py`, this script
will also generate the html files for the conversations, produce a csv that contains the
relevant results for the experiments (set the experiment group name at the top of the
file). Finally it will also render plots, I put the scatter animation in a separate
script `animate_scatter.py` but it requires the csv produced by `quantitative_plots.py`.
If you want to produce a csv you a custom set of runs use the first command below:
```bash
# (optional) Generate csvs for quantitative properties of certain runs.
python analyse_quantitative.py --output quantitative_metrics.csv --runs 34 45 54 ...

# This script will actually render the dashboard plots and html files for a for the appropriate set of experiments in `completed_runs.json` and you must set the experiment group name at the top of the script. It also runs the `analyse_quantitative.py` with all of the appropriate set of experiments.
python quantitative_plots.py

# This script should be run after, it requires the table produced by `quantitative_plots.py`
python animate_scatter.py
```

---

## Using LLMs to Analyse the Conversations


```bash
python analyse_qualitative.py \
    --input test_cases/conversations/frankenstein_run_83_member_1_conv_2.json \
    --model openai/gpt-5-mini \
    --out-file test_cases/model_evaluations/gpt_5_mini/eval_${i}_frankenstein_run_83_member_1_conv_2.json
```
This script analyses the conversation which we manually augmented to contain all cases
of deception. It produces a plot comparing different models at how well they analyse the
deception schema relative to a reference model.
```bash
python -m test_cases.model_evaluations.evaluate_consistency
```
The following script has three stages, you must select which in the `if __name__ ==
"__main__":` section. The first stage runs the sentiment analysis with LLMs over each
conversation. The second stage checks all analysis runs completed and re-runs the ones
which fails, when this section succeeds you can move onto stage 3. Stage 3 plots all of
the data based on the deception analysis outputs from the LLM.
```bash
python finalise_qualitative.py
```
This scripts make a simple plot to show how many of each model type was flagged for some
part of the deception criteria.
```bash
python interesting_runs_report.py
```


---

## Natural Deception Runs

The natural deception runs take a simulation where an agent naturally started acting
deceptively and copy it. But the copy is only done up to a certain round, this means
that we can resume it from different points and see how often it doubles down.
Additionally, you can optionally add snitch agents which modifies the config, such that
when the runs are resumed the snitches will be instantiated. You can also override the
number of rounds.
```bash
python replicate_intermediate_run_state.py \
    --trim-round 5 \
    --run-index 85 \
    --add_snitch_models openai/gpt-5-mini openai/gpt-5-mini openai/gpt-5-mini \
        openai/gpt-5-mini \
    --num-rounds 10
```
you can modify the `replicate_run85` inside with your own run, just as a helper.


---

## Snitch Agents

The following script is based on the `simulate_cpr.py` script. It has a superset of the
functionality, but the script is longer and a bit more complicated.
```bash
# Resume a checkpoint, e.g. a replicated run as per the last section
python snitch_cpr.py --seed 3 --run-index 207

# Run with a saboteur
python snitch_cpr.py \
    --seed 0 \
    --agent-models moonshotai/kimi-k2-thinking openai/gpt-5 openai/gpt-5 openai/gpt-5 \
    --snitch-models openai/gpt-5-mini openai/gpt-5-mini openai/gpt-5-mini openai/gpt-5-mini \
    --saboteur Member1
```
