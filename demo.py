"""Compatibility entry point for EgoHSI3R inference."""
from infer import parse_args, run_inference

if __name__ == "__main__":
    run_inference(parse_args())
