"""Resolve and validate an explicit DRAFT configuration; never launch GPUs."""
import argparse
import sys

from config_contract import ROOT, atomic_json, fingerprint, production_blockers, resolve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=str, required=True)
    parser.add_argument("--base", type=str)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--production", action="store_true")
    args = parser.parse_args()
    try:
        cfg = resolve(args.profile, args.base)
        blockers = production_blockers(cfg)
        if args.production:
            raise ValueError("production launch is blocked: " + "; ".join(blockers))
        envelope = {"status": "draft_validated", "production_ready": False,
                    "config_sha256": fingerprint(cfg), "production_blockers": blockers,
                    "config": cfg}
        atomic_json(args.output, envelope)
        print(f"Draft validated: {cfg['run']['arm']}; production_ready=false; {len(blockers)} blockers.")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
