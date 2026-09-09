import argparse
import shutil

import mlflow.artifacts

from src.registry import REGISTERED_MODEL_NAME, CHAMPION_ALIAS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="deploy/model",
                    help="Where to write the exported model directory.")
    p.add_argument("--version", default=None,
                    help="Export a specific registered version number instead of the current champion alias.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.version:
        model_uri = f"models:/{REGISTERED_MODEL_NAME}/{args.version}"
    else:
        model_uri = f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}"

    print(f"[Export] Fetching: {model_uri}")

    shutil.rmtree(args.output, ignore_errors=True)

    try:
        dst = mlflow.artifacts.download_artifacts(
            artifact_uri=model_uri,
            dst_path=args.output,
        )
    except Exception as e:
        print(f"[Export] FAILED — is there a champion registered yet? ({e})")
        print("[Export] Run main.py at least once first so a champion exists.")
        raise SystemExit(1)

    print(f"[Export] Exported to: {dst}")


if __name__ == "__main__":
    main()