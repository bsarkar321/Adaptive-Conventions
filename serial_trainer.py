import os
from config import get_config

from overcooked_env.overcooked_env import PantheonOvercooked
from XD.serial import run_serial


def generate_gym(args):
    """Generate the gym given the command-line arguments."""
    if args.env_name == "Overcooked":
        args.hanabi_name = "overcooked"
        return PantheonOvercooked(args.over_layout)
    return None


def main():
    args = get_config().parse_args()
    print(args)
    pop_size = args.pop_size
    env = generate_gym(args)
    run_dir = (
        os.path.dirname(os.path.abspath(__file__))
        + "/"
        + args.hanabi_name
        + "_models/"
        + str(args.seed)
    )
    os.makedirs(run_dir, exist_ok=True)
    with open(run_dir + "/" + "args.txt", "w", encoding="UTF-8") as file:
        file.write(str(args))

    run_serial(pop_size, args, env, run_dir, "cpu", restored=args.restored)


if __name__ == "__main__":
    main()
