"""CLI:
  python -m argus.training cache --root datasets/raw --out datasets/features
  python -m argus.training train --features datasets/features --out checkpoints/bc_v1.pt
"""
import argparse


def main():
    p = argparse.ArgumentParser(prog="argus.training")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cache")
    c.add_argument("--root", default="datasets/raw")
    c.add_argument("--out", default="datasets/features")
    c.add_argument("--no-strict", action="store_true", help="also cache episodes that fail the audit")
    c.add_argument("--fake", action="store_true", help="pipeline smoke test with random features")

    t = sub.add_parser("train")
    t.add_argument("--features", nargs="+", default=["datasets/features"],
                   help="one or more feature folders, e.g. datasets/features datasets/features_d1")
    t.add_argument("--out", default="checkpoints/bc_v1.pt")
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--bs", type=int, default=16)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--history", type=int, default=6)
    t.add_argument("--holdout-task", default=None, help="train without this task, evaluate on it")
    t.add_argument("--seed", type=int, default=0)

    a = p.parse_args()
    if a.cmd == "cache":
        from argus.training.cache_features import cache_features

        if a.fake:
            from argus.perception.vision_encoder import RandomVisionEncoder

            v = RandomVisionEncoder()
        else:
            from argus.perception.vision_encoder import VisionEncoder

            print("loading DINOv2-small (first run downloads the weights)...")
            v = VisionEncoder()
        cache_features(a.root, a.out, v, strict=not a.no_strict)
    else:
        from argus.training.behavioral_cloning import train

        train(a.features[0] if len(a.features) == 1 else a.features, a.out, epochs=a.epochs, bs=a.bs, lr=a.lr, T=a.history,
              holdout_task=a.holdout_task, seed=a.seed)


if __name__ == "__main__":
    main()
