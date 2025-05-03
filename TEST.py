import os
import json

# Mapping of old env names to new ones
ENV_MAP = {
    "BattleZone": "Amidar",
    "NameThisGame": "Frostbite",
    "Phoenix": "Riverraid",
    "Qbert": "KungFuMaster"
}

def replace_env_names_in_command(command: str) -> str:
    for old, new in ENV_MAP.items():
        command = command.replace(f"{old}-v5", f"{new}-v5")
    return command

def process_json_file(filepath: str):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Failed to read {filepath}: {e}")
        return

    modified = False
    if isinstance(data, dict) and "experiments" in data:
        for exp in data["experiments"]:
            if "command" in exp:
                new_command = replace_env_names_in_command(exp["command"])
                if new_command != exp["command"]:
                    exp["command"] = new_command
                    modified = True

    if modified:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Updated: {filepath}")
    else:
        print(f"No changes in: {filepath}")

def walk_and_process_jsons(root_dir: str):
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".json"):
                full_path = os.path.join(dirpath, file)
                process_json_file(full_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Replace env names in JSON experiment commands.")
    parser.add_argument("path", type=str, help="Path to root directory containing JSON files.")

    args = parser.parse_args()
    walk_and_process_jsons(args.path)
