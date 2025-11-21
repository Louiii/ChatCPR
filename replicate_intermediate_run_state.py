"""
python replicate_intermediate_run_state.py --trim-round 5 --run-index 85 \
    --add_snitch_models openai/gpt-5-mini openai/gpt-5-mini openai/gpt-5-mini \
        openai/gpt-5-mini \
    --num-rounds 10
"""
import argparse
import json
from pathlib import Path

from snitch_cpr import SNITCH_PROMPT


def copy_and_trim_run(
    trim_round: int,
    copy_run_index: int,
    add_snitch_models: list[str] = None,
    num_rounds: int | None = None,
    directory: str = "runs",
    log_name: str = "main_log.json",
):
    output_dir = Path(directory)

    run_dir = output_dir / f"run_{copy_run_index}"
    config = json.loads((run_dir / "config.json").read_text())
    main_log = json.loads((run_dir / log_name).read_text())

    # Setup the target directory
    runs = [int(p.name.split("_")[-1]) for p in output_dir.glob('*')]
    new_run_index = (max(runs) + 1) if len(runs) else 1
    new_run_dir = output_dir / f"run_{new_run_index}"
    (new_run_dir / "states").mkdir(parents=True, exist_ok=True)

    # Modify config
    if add_snitch_models:
        config["snitch_models"] = add_snitch_models
    if num_rounds is not None:
        config["simulation_rounds"] = num_rounds

    # Write the trimmed/modified files.
    (new_run_dir / "config.json").write_text(json.dumps(config))
    (new_run_dir / log_name).write_text(json.dumps(main_log[:trim_round]))
    for i in range(1, trim_round + 1):
        name = f"states/round_{i}.json"
        (new_run_dir / name).write_text((run_dir / name).read_text())
    for path in run_dir.glob("*.log"):
        (new_run_dir / path.name).write_text(path.read_text())
    for member_dir in run_dir.glob("Member*"):
        new_member_dir = new_run_dir / member_dir.name
        new_member_dir.mkdir(parents=True, exist_ok=True)
        for conv_file in member_dir.glob("conversation_*.json"):
            start, stop = [int(i) for i in conv_file.name.split("_")[3].split(".")[0].split("-")]
            if start <= trim_round:
                conv = json.loads(conv_file.read_text())
                if add_snitch_models:
                    conv[0]["guard_prompt"] = SNITCH_PROMPT
                    conv[0]["guard_model"] = add_snitch_models[int(member_dir.name.split("Member")[-1]) - 1]
                if stop > trim_round:
                    # We need to crop the conversation.
                    start_ixs = [
                        i for i, c in enumerate(conv)
                        if c["phase"] == 'GENERAL_DISCUSSION: outgoing 0' and c["role"] == "user"
                    ]
                    assert start_ixs[0] == 1
                    we_need = stop - trim_round
                    end = [i for i in start_ixs][we_need]
                    conv = conv[:end]
                (new_member_dir / conv_file.name).write_text(json.dumps(conv))
    return new_run_dir


def main():
    parser = argparse.ArgumentParser(description="Run CPR simulation")
    parser.add_argument(
        "--run-index",
        type=int,
        help="Run index to continue from (creates new run if not specified)"
    )
    parser.add_argument(
        "--directory",
        type=str,
        default="runs",
        help="Directory for runs (default: runs)"
    )
    parser.add_argument(
        "--trim-round",
        type=int,
        default=1,
        help="Random seed (default: 1)"
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=1,
        help="Random seed (default: 1)"
    )
    parser.add_argument(
        "--add-snitch-models",
        nargs="+",
        default=None,
        help="List of agent models to use as snitches, must match the length agent models"
    )
    args = parser.parse_args()

    copy_and_trim_run(
        trim_round=args.trim_round,
        copy_run_index=args.run_index,
        add_snitch_models=args.snitch_models,
        num_rounds=args.num_rounds,
        directory=args.directory,
    )


def replicate_run85():
    from time import sleep
    
    run_index = 85

    trim_rounds = [1, 3, 5]
    snitch_models = [None, ["openai/gpt-5-mini"]*4]

    for sm in snitch_models:
        for tr in trim_rounds:
            new_run_dir = copy_and_trim_run(
                trim_round=tr,
                copy_run_index=run_index,
                add_snitch_models=sm,
                num_rounds=10,
            )
            print(f"cp run{run_index} -> setup snitches: {sm}, trimmed rounds to {tr} -> {new_run_dir}")
            sleep(10)


if __name__ == "__main__":
    main()
    # replicate_run85()
#