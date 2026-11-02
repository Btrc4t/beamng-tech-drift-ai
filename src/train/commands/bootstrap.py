"""Make an offline (BC) model ready to continue online.

``train.py bootstrap`` loads the BC model (which also verifies it is the
requested algorithm) and writes it as ``last_model.zip`` into a fresh online
run dir under ``models/<ALGO>/``. A subsequent ``online --algo <algo>`` then
auto-resumes from that checkpoint, so the online run continues from the
cloned weights instead of training from scratch. This is the no-``--init``
way to do the BC -> online warm start.
"""
from datetime import datetime
from pathlib import Path

from train.common import ALGOS, ResumeFrom, _latest_bc_model


def cmd_bootstrap(args) -> None:
    MODEL, _ = ALGOS[args.algo]
    model_path = (Path(args.model) if args.model
                  else _latest_bc_model(Path("models"), args.algo))
    if model_path is None or not model_path.is_file():
        raise SystemExit(
            f"no BC model for '{args.algo}' under models/bc/{args.algo}/; "
            f"train one first (offline --algo {args.algo}) or pass --model")
    loaded = MODEL.load(model_path, verbose=0)
    run_dir = Path(f"models/{MODEL.__name__}/"
                   f"{datetime.now().strftime('%Y.%m.%d-%H.%M')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / ResumeFrom.LAST.filename
    loaded.save(out)
    print(f"loaded {model_path} ({MODEL.__name__}, "
          f"num_timesteps={loaded.num_timesteps})")
    print(f"bootstrapped -> {out}")
    print(f"continue with: python train.py online --algo {args.algo}")
