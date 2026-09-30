import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, dt_ms):
        # dt_ms: (B, 1) in milliseconds
        device = dt_ms.device
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(0, half_dim, dtype=torch.float32, device=device) / half_dim
        )
        args = dt_ms * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UpConvBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class AnticipatoryMotionSuppressionNet(nn.Module):
    """
    Anticipatory Motion Suppression Network for Event Cameras (UZH RSS 2026 inspired).
    Takes a 2-bin temporal event stack (B, 2, H, W) and future horizon dt_p (B, 1).
    Outputs:
      1. warped_mask (B, 1, H, W): Zero-latency dynamic object mask (IMO = 1, Background = 0).
      2. optical_flow (B, 2, H, W): Predicted forward optical flow (u, v) in pixels.
    """
    def __init__(self, in_channels=2, base_ch=32):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(in_channels, base_ch, stride=1)      # (B, 32, H, W)
        self.enc2 = ConvBlock(base_ch, base_ch * 2, stride=2)      # (B, 64, H/2, W/2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4, stride=2)  # (B, 128, H/4, W/4)

        # Attention-based Time Conditioning (ATC)
        self.pe = SinusoidalPositionalEncoding(base_ch * 4)
        self.time_mlp = nn.Sequential(
            nn.Linear(base_ch * 4, base_ch * 4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(base_ch * 4, base_ch * 4 * 2) # Scale + Shift
        )

        # Mask Decoder (DM)
        self.mask_up2 = UpConvBlock(base_ch * 4, base_ch * 2, base_ch * 2)
        self.mask_up1 = UpConvBlock(base_ch * 2, base_ch, base_ch)
        self.mask_out = nn.Sequential(
            nn.Conv2d(base_ch, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # Optical Flow Decoder (D_psi)
        self.flow_up2 = UpConvBlock(base_ch * 4, base_ch * 2, base_ch * 2)
        self.flow_up1 = UpConvBlock(base_ch * 2, base_ch, base_ch)
        self.flow_out = nn.Conv2d(base_ch, 2, kernel_size=3, padding=1)

    def forward(self, events, dt_p):
        # 1. Spatio-Temporal Encoding
        e1 = self.enc1(events)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # 2. Time-Conditioning for Flow Decoder
        t_embed = self.pe(dt_p) # (B, 128)
        scale_shift = self.time_mlp(t_embed).unsqueeze(-1).unsqueeze(-1) # (B, 256, 1, 1)
        scale, shift = torch.chunk(scale_shift, 2, dim=1) # each (B, 128, 1, 1)
        e3_conditioned = e3 * (1.0 + torch.tanh(scale)) + shift

        # 3. Decode Current IMO Mask
        m2 = self.mask_up2(e3, e2)
        m1 = self.mask_up1(m2, e1)
        mask = self.mask_out(m1) # (B, 1, H, W)

        # 4. Decode Future Optical Flow
        f2 = self.flow_up2(e3_conditioned, e2)
        f1 = self.flow_up1(f2, e1)
        flow = self.flow_out(f1) # (B, 2, H, W)

        # 5. Differentiable Backward Flow Warping
        # Grid creation
        B, _, H, W = events.shape
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=events.device),
            torch.linspace(-1.0, 1.0, W, device=events.device),
            indexing='ij'
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1) # (B, H, W, 2)

        # Normalize optical flow displacements from pixels to [-1, 1] normalized coordinates
        flow_norm_x = flow[:, 0, :, :] * (2.0 / max(1.0, W - 1))
        flow_norm_y = flow[:, 1, :, :] * (2.0 / max(1.0, H - 1))
        flow_norm = torch.stack([flow_norm_x, flow_norm_y], dim=-1)

        # Sampling grid: warp backward x_sample = x - flow
        sample_grid = base_grid - flow_norm

        # Bilinear interpolation
        warped_mask = F.grid_sample(mask, sample_grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        return warped_mask, flow

def main():
    os.makedirs("models", exist_ok=True)
    model = AnticipatoryMotionSuppressionNet(in_channels=2, base_ch=32)
    model.eval()

    # Resolution: 360 x 640 (Subsampled 2x from 720x1280 IMX636 for real-time 200 Hz inference on Orin Nano)
    dummy_events = torch.randn(1, 2, 360, 640, dtype=torch.float32)
    dummy_dt = torch.tensor([[40.0]], dtype=torch.float32) # 40ms forecast

    onnx_path = "models/event_suppression.onnx"
    print(f"Exporting model to {onnx_path}...")
    torch.onnx.export(
        model,
        (dummy_events, dummy_dt),
        onnx_path,
        input_names=["events_tensor", "forecast_dt_ms"],
        output_names=["warped_mask", "optical_flow"],
        opset_version=17,
        dynamo=False,
        dynamic_axes={
            "events_tensor": {0: "batch_size"},
            "forecast_dt_ms": {0: "batch_size"},
            "warped_mask": {0: "batch_size"},
            "optical_flow": {0: "batch_size"}
        }
    )
    print(f"ONNX Export successful! Size: {os.path.getsize(onnx_path) / 1024:.1f} KB")

if __name__ == "__main__":
    main()
