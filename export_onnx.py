import os
import torch
from train_spectral_combnet import SpectralCombNet

def export_single():
    model = SpectralCombNet(in_bins=257, base_channels=32)
    state = torch.load("spectral_combnet_best.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 1, 257, dtype=torch.float32)
    output_path = "spectral_combnet.onnx"

    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["spectrum_in"],
        output_names=["drone_prob", "fund_freq_hz", "harmonic_purity"],
        dynamic_axes={
            "spectrum_in": {0: "batch_size"},
            "drone_prob": {0: "batch_size"},
            "fund_freq_hz": {0: "batch_size"},
            "harmonic_purity": {0: "batch_size"}
        },
        dynamo=False
    )
    print(f"Exported self-contained ONNX to {output_path} ({os.path.getsize(output_path):,} bytes)")

if __name__ == "__main__":
    export_single()
