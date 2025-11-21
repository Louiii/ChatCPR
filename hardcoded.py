import json
from collections import namedtuple
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

from simulate_cpr import (
    Agent,
    AnswerCOT,
    Config,
    ConversationEvent,
    WorldState,
    run_round,
)

DIRECTORY = "hardcoded_runs"


Conv = namedtuple("Conv", ["id"])
DummyClient = namedtuple("DummyClient", ["responses", "conversations"])


class Conversation:
    def create(self, *args, **kwargs):
        name = kwargs["metadata"]["topic"].replace("CPR/", "")
        return Conv(id=name)


class DummyChat:
    def __init__(
        self,
        name,
        all_agents,
        policy_str,
        extraction_amount,
        num_messages: int = 0,
        send_public_message: bool = True,
    ):
        self.name = name
        self.all_agents = all_agents
        self.policy_str = policy_str
        self.num_messages = num_messages
        self.send_public_message = send_public_message
        self.nonempty_msg = "blah " * 3
        self.extraction_amount = extraction_amount

    def ask(self, user_input, phase):
        if ConversationEvent.GENERAL_DISCUSSION.name in phase:
            others = [a for a in self.all_agents if a != self.name]
            ans = json.dumps({
                "dms": [
                    {"to": rec, "msg": self.nonempty_msg}
                    for rec in others[:self.num_messages]
                ],
                "public_message": self.nonempty_msg if self.send_public_message else "",
            })
        elif ConversationEvent.AUDIT_VOTE.name == phase:
            ans = self.policy_str
        elif ConversationEvent.RESOURCE_EXTRACTION.name == phase:
            x = str(self.extraction_amount)
            ans = "{\n  \"extract_true\": " + x + ",\n  \"extract_reported\": " + x + "\n}"
        elif ConversationEvent.TRIGGER_ACCUSATION.name == phase:
            ans = "{\n  \"accuse\": false\n}"
        elif ConversationEvent.REPRODUCTION_OPTION.name == phase:
            ans = "{\n  \"spawn\": false\n}"
        else:
            # Only last case is summarisation responses.
            ans = ""
        return AnswerCOT(ans, "")


def simulate(
    num_agents: int,
    num_private_dms: int,
    send_public_message: bool,
    quota: float,
    run_index: int | None = None,
    directory: str = "runs",
    save: bool = True,
    log_name: str = "main_log.json",
    seed: int = 0,
    agent_models: list[str] | None = None,
):
    np.random.seed(seed)

    # Create directory for new run.
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dir = output_dir / f"run_{run_index}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    main_log_path = run_dir / log_name

    config = Config(agent_models)

    if save:
        # Save config
        config_path.write_text(json.dumps(config.__dict__))

    world_state = WorldState(
        R=config.env_starting_resource,
        measurement_std=0.5,
        threshold=config.threshold,
        quota=config.quota,
        audit_probability=config.audit_probability,
        penalty_multiplier=config.penalty_multiplier,
        base_penalty=config.base_penalty,
    )
    
    all_agents = [f"Member{i}" for i in range(1, 1 + num_agents)]
    policy = "{\n  \"threshold\": 10.0,\n  \"quota\": "+str(quota)+",\n  \"audit_probability\": 0.5,\n  \"penalty_multiplier\": 3.0,\n  \"base_penalty\": 1.0\n}"

    # Only considering fair agents atm.
    extraction_amount = quota / num_agents

    agents = []
    for i in enumerate(range(num_agents), start=1):
        name = f"Member{i}"
        agents.append(
            Agent(
                name=name,
                credits=config.initial_credits,
                conversation=DummyChat(
                    name,
                    all_agents,
                    policy,
                    extraction_amount=extraction_amount,
                    num_messages=num_private_dms,
                    send_public_message=send_public_message,
                ),
                conversation_round_count=0,
                conversation_count=1,
            )
        )

    next_context = {}
    dead_agents = []
    all_round_logs = []
    resource_estimate_history = []  # Just to give context to the agents
    start_round = 0

    for round_idx in range(start_round, config.simulation_rounds):
        # Run round.
        print("Credits: " + ", ".join([f"{a.name}: {a.credits:.2f}" for a in agents]))
        world_state, agents, dead_agents, depleted_resource, round_log, next_context = run_round(
            world_state=world_state,
            agents=agents,
            dead_agents=dead_agents,
            round_index=round_idx + 1,
            config=config,
            next_context=next_context,
            resource_estimate_history=resource_estimate_history,
        )
        all_round_logs.append(round_log)
        main_log_path.write_text(json.dumps(all_round_logs))
        print(f"\n{'-'*80}\nRound {round_idx + 1} complete.\n{'-'*80}\n")
        if depleted_resource:
            break


def get_runs(directory):
    return [int(p.name.split("_")[-1]) for p in directory.glob("run_*") if p.is_dir()]


def run_experiment(
    num_runs: int,
    num_agents: int,
    num_private_dms: int,
    send_public_message: bool,
    quota: float,
    note: str,
):
    directory = Path(DIRECTORY)
    runs = get_runs(directory)
    run_index = (max(runs) + 1) if len(runs) else 1

    for seed in range(num_runs):
        simulate(
            run_index=run_index,
            directory=directory,
            num_agents=num_agents,
            num_private_dms=num_private_dms,
            send_public_message=send_public_message,
            quota=quota,
            log_name=f"main_log_seed_{seed}.json",
            seed=seed,
            # num_starting_agents=num_agents,
        )

    text = f"{note}\n{num_agents=}\n{quota=}\n{num_private_dms=}\n"
    run_index = max(get_runs(directory))
    (directory / f"run_{run_index}/note.txt").write_text(text)
    return run_index


def fair_share_minimal_cost(cfg, num_private_dms: int, public_msg: bool):
    _conv_cost = (cfg.cost_pub * int(public_msg) + cfg.cost_dm * num_private_dms) * 2
    _conv1_cost = cfg.num_rounds_initial_conversation * _conv_cost
    round_cost = _conv1_cost + cfg.maintenance_tax
    required_per_agent_extraction = round_cost / cfg.price_per_unit
    return required_per_agent_extraction


def main():
    num_runs = 20
    cfg = Config(None)

    # Constant config.
    num_agents = 4

    def _stack_path(num_agents: int):
        return Path(DIRECTORY) / f"exp_stack_num_agents_{num_agents}.json"

    def run_exps(num_agents: int):
        """Experiments for different messaging behaviours.

        stage 0 = no msgs
        stage 1 = only pub msg
        stage 2 = pub + pri
        ...
        stage n = pub + (n-1) * pri
        
        n-1 == num_agents-1 == max possible private messages.
        """
        info = []
        for stage in range(num_agents + 1):
            num_private_dms = max(0, stage - 1)
            send_public_message = bool(stage > 0)
            num_msgs = num_private_dms + int(send_public_message)
            quota = num_agents * fair_share_minimal_cost(
                cfg, num_private_dms=num_private_dms, public_msg=send_public_message
            )
            run_index = run_experiment(
                num_runs=num_runs,
                num_agents=num_agents,
                num_private_dms=num_private_dms,
                send_public_message=send_public_message,
                quota=quota,
                note=f"Fair share at minimal survival cost.\n- Message {num_msgs}\n- No accusations\n- No spawning",
            )
            info.append(
                {
                    "run": run_index,
                    "num_agents": num_agents,
                    "num_messages": num_msgs,
                    "quota": quota,
                }
            )
        return info 


    num_agent_stacks = list(range(2, 8))
    for num_agents in num_agent_stacks:
        _stack_path(num_agents).write_text(json.dumps(run_exps(num_agents)))

    all_runs = [json.loads(_stack_path(n).read_text()) for n in num_agent_stacks]
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8), sharey=True)
    for ax, runs, num_agents in zip(axes.flatten(), all_runs, num_agent_stacks):
        cols = sns.color_palette(palette="rocket", n_colors=len(runs))
        legend_handles = []
        for c, run in zip(cols, runs):
            run_path = Path(DIRECTORY) / f"run_{run['run']}"
            for i, seed_path in enumerate(run_path.glob("main_log_seed_*.json")):
                data = json.loads(seed_path.read_text())
                resource = np.array([item["world_state_start"]["R"] for item in data])
                label = f"[{run['num_messages']}; {run['quota']:.1f}]"
                if i==0:
                    legend_handles.append(Line2D([0], [0],  color=c,  linewidth=2.0, label=label))
                ax.plot(resource, c=c, lw=0.5)
        ax.legend(title="$[N_{\\text{msg}}, Q]$", handles=legend_handles, fontsize=9)
        ax.text(0.5, 0.92, f"{num_agents} Agents",
            transform=ax.transAxes,
            ha='center', va='center', fontsize=13)
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, hspace=0.0, wspace=0.0)
    plt.savefig("fair_share_min_cost_sims.png")
    plt.close(fig)


if __name__ == "__main__":
    main()