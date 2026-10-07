from real_spectra_dataset import RealSpectraDataset, CombinedSpectralDataset
import torch

ds_real = RealSpectraDataset("real_darkroom_spectra.bin")
print("Real dataset count:", len(ds_real))
x, y, f0, purity = ds_real[0]
print("Real sample 0 shape:", x.shape, "label:", y.item(), "max val:", x.max().item())

ds_comb = CombinedSpectralDataset(num_samples=100, real_spectra_path="real_darkroom_spectra.bin")
print("Combined dataset count:", len(ds_comb))
x_c, y_c, f0_c, p_c = ds_comb[0]
print("Sample combined shape:", x_c.shape, "label:", y_c.item(), "f0:", f0_c.item())
print("Real spectra dataset verification passed!")
