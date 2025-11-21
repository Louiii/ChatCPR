import os
import subprocess
from pathlib import Path
from time import sleep

from replicate_intermediate_run_state import replicate_run85


def load_secrets(filepath=".secrets"):
    env = os.environ.copy()  # Start with current environment
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):  # Skip empty lines and comments
                key, value = line.split("=", 1)
                env[key.strip()] = value.strip()
    return env


ENV = load_secrets()
PYTHON_BIN = "python"


seeds = [0, 1, 2, 3, 4]
# sleep(60 * 60 * 4)


def get_run_index():
    return max(int(p.name.split("_")[-1]) for p in Path(f"runs/").glob("run_*")) + 1


def launch(exps, script="simulate_cpr.py", pause=60):
    processes = []
    ri = get_run_index()
    for seed in seeds:
        for exp_name, args in exps.items():
            cmd = [PYTHON_BIN, script, "--seed", str(seed), "--agent-models"] + args

            # Open log file for this run
            log_file = open(f"output{ri}.log", "w")

            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,  # Redirect stderr to stdout (equivalent to 2>&1)
                env=ENV,
            )
            processes.append((process, log_file))
            print(f"Started: {exp_name} -> {ri} | {seed=} (logging to output{ri}.log)")
            sleep(20)
            ri += 1
        sleep(pause)

    # Wait for all processes to complete and close log files
    for process, log_file in processes:
        process.wait()
        log_file.close()

    print("All simulations complete!")


def resume_runs(run_indices_and_seeds, script="snitch_cpr.py", pause=60):
    processes = []
    for ri, seed in run_indices_and_seeds:
        cmd = [PYTHON_BIN, script, "--seed", str(seed), "--run-index", str(ri)]
        log_file = open(f"output{ri}.log", "w")
        process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=ENV)
        processes.append((process, log_file))
        print(f"Started: {ri} | {seed=} (logging to output{ri}.log)")
        sleep(pause)
    # Wait for all processes to complete and close log files
    for process, log_file in processes:
        process.wait()
        log_file.close()


def regular_runs():
    agents = {
        # Speedy and works pretty well. [Cost of 5 runs: ~$0.20]
        "Homogenous GPT5-Nano": ["openai/gpt-5-nano"] * 4,
        # Decent speed and works well. [Cost of 5 runs: ~$1.00]
        "Homogenous GPT5-Mini": ["openai/gpt-5-mini"] * 4,
        # Slowish but seems to work well. [Cost of 5 runs: ~$5.00]
        "Homogenous GPT5": ["openai/gpt-5"] * 4,
        # Buggy! Others have reported not being able to use it in the API
        "Homogenous Gemini": ["gemini/gemini-2.5-flash"] * 4,
        # # This one doesn't work that well, often not writing valid json.
        # agents = ["gemini/gemini-2.5-flash-lite"] * 4,
        # Unbelievably slow. [Cost of 5 runs: ~$1.00]
        "Homogenous DeepSeek": ["deepseek/deepseek-reasoner"] * 4,
        # Unbelievably expensive. [Cost of 5 runs: ~$30.00]
        "Homogenous Claude": ["anthropic/claude-haiku-4-5"] * 4,
        # Kimi-K2-Thinking
        "Homogenous Kimi": ["moonshotai/kimi-k2-thinking"] * 4,
        "Hetrogenous Companies": [
            "gemini/gemini-2.5-flash", "deepseek/deepseek-reasoner", "openai/gpt-5-mini", "anthropic/claude-haiku-4-5",
        ],
        "Hetrogenous Intelligence": [
            "gemini/gemini-2.5-flash-lite", "openai/gpt-5-nano", "openai/gpt-5-mini", "openai/gpt-5",
        ],
    }
    launch(agents)


def saboteur_runs():
    g,k,d,c = "openai/gpt-5", "moonshotai/kimi-k2-thinking", "deepseek/deepseek-reasoner", "anthropic/claude-haiku-4-5"
    agents = {
        "GPT-5 Saboteur": [g,k,d,c],
        "Kimi Saboteur": [k,d,c,g],
        "DeepSeek Saboteur": [d,c,g,k],
        "Claude Saboteur": [c,g,k,d],
    }
    launch({k: v + ["--saboteur", "Member1"] for k, v in agents.items()})


def saboteur_with_snitches_runs():
    launch(
        exps={
            "Kimi Saboteur; GPT-5 Mini Snitches": [
                "--agent-models",
                "moonshotai/kimi-k2-thinking",
                "openai/gpt-5",
                "deepseek/deepseek-reasoner",
                "anthropic/claude-haiku-4-5",
            ] + ["--saboteur", "Member1", "--snitch-models"] + ["openai/gpt-5-mini"]*4,
        },
        script="snitch_cpr.py",
    )


def natural_deception_runs():
    for seed in seeds:
        start_ix = get_run_index()
        # Setup the runs, this copies out naturally deceptive run into six new experiments
        replicate_run85()
        resume_runs([(ri, seed) for ri in range(start_ix, start_ix + 6)])


def natural_snitches_runs():
    a = ["--agent-models"] + ["moonshotai/kimi-k2-thinking"]*4
    launch(
        exps={
            "Kimi; GPT-5 Mini Snitches": a + ["--snitch-models"] + ["openai/gpt-5-mini"]*4,
            "Kimi; No snitches": a,
        },
        script="snitch_cpr.py",
    )


if __name__ == "__main__":
    regular_runs()
    saboteur_runs()
    saboteur_with_snitches_runs()
    natural_deception_runs()
    natural_snitches_runs()