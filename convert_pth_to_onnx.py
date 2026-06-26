import argparse
import re
from pathlib import Path

import onnx
import torch

from DQNAgent import DQN


def parse_linear_layers(state_dict):
    """Extract linear layer sizes from keys like fc.0.weight, fc.2.weight, ..."""
    pattern = re.compile(r"^fc\.(\d+)\.weight$")
    linear_layers = []

    for key, tensor in state_dict.items():
        match = pattern.match(key)
        if match:
            idx = int(match.group(1))
            out_features, in_features = tensor.shape
            linear_layers.append((idx, in_features, out_features))

    if not linear_layers:
        raise ValueError("Could not infer network architecture from state_dict (no fc.*.weight keys found).")

    linear_layers.sort(key=lambda x: x[0])

    input_dim = linear_layers[0][1]
    output_dim = linear_layers[-1][2]
    hidden_layers = [layer[2] for layer in linear_layers[:-1]]

    return input_dim, output_dim, hidden_layers


def add_metadata_to_model(onnx_path, pth_path):
    model = onnx.load(str(onnx_path))

    source_meta = model.metadata_props.add()
    source_meta.key = "source_pth"
    source_meta.value = str(pth_path)

    model_name_meta = model.metadata_props.add()
    model_name_meta.key = "model_name"
    model_name_meta.value = pth_path.stem

    onnx.save(model, str(onnx_path))


def convert_file(pth_path, overwrite=False):
    pth_path = Path(pth_path)
    onnx_path = pth_path.with_suffix(".onnx")

    if onnx_path.exists() and not overwrite:
        print(f"[SKIP] ONNX already exists: {onnx_path}")
        return "skipped"

    state_dict = torch.load(str(pth_path), map_location=torch.device("cpu"))
    input_dim, output_dim, hidden_layers = parse_linear_layers(state_dict)

    model = DQN(input_dim=input_dim, output_dim=output_dim, hidden_layers=hidden_layers, device=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.zeros(1, input_dim, dtype=torch.float32)

    torch.onnx.export(
        model,
        (dummy_input,),
        str(onnx_path),
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
    )

    add_metadata_to_model(onnx_path, pth_path)
    print(f"[OK] Converted: {pth_path} -> {onnx_path}")
    return "converted"


def main():
    parser = argparse.ArgumentParser(
        description="Convert all .pth DQN checkpoints in a folder (recursively) to .onnx files."
    )
    parser.add_argument(
        "--models-dir",
        default="trained_agents",
        help="Directory containing .pth files (default: trained_agents)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .onnx files",
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    if not models_dir.exists() or not models_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {models_dir}")

    pth_files = sorted(models_dir.rglob("*.pth"))
    if not pth_files:
        print(f"No .pth files found under: {models_dir}")
        return

    converted = 0
    skipped = 0
    failed = 0

    for pth_path in pth_files:
        try:
            result = convert_file(pth_path, overwrite=args.overwrite)
            if result == "converted":
                converted += 1
            else:
                skipped += 1
        except Exception as exc:
            failed += 1
            print(f"[ERROR] Failed to convert {pth_path}: {exc}")

    print("\nConversion summary:")
    print(f"  Converted: {converted}")
    print(f"  Skipped:   {skipped}")
    print(f"  Failed:    {failed}")


if __name__ == "__main__":
    main()

