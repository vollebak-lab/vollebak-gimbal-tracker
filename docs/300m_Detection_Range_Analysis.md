# 300m Detection Range Analysis — GenX320 vs DVXplorer Micro

## Key Insight: Temporal Detection ≠ Spatial Resolution

Event cameras detect **temporal brightness modulation** (propeller flicker), not spatial shape. The real question: at what range does the drone body (~30cm) subtend at least 1 pixel?

## Range Calculations

### With 12mm M12 lens

| Sensor | Pixels on drone @ 300m | 1-pixel limit | FOV |
| -- | -- | -- | -- |
| GenX320 (6.3µm) | **1.9** ✅ | 571m | 9.6° |
| DVXplorer Micro (9µm) | 1.33 (marginal) | 400m | 27.5° |

### With 16mm M12 lens

| Sensor | Pixels on drone @ 300m | 1-pixel limit | FOV |
| -- | -- | -- | -- |
| GenX320 | **2.54** ✅ | 762m | 7.2° |
| DVXplorer Micro | **1.78** ✅ | 533m | 20.6° |

### With 6mm M12 lens (selected for Predator)

| Sensor | Pixels on drone @ 300m | 1-pixel limit | FOV |
| -- | -- | -- | -- |
| GenX320 | 0.95 (sub-pixel) | 286m | 19.2° |
| DVXplorer Micro | 0.67 (not viable) | **200m** | **55°** |

## Recommended Architecture (Selected)

**4× DVXplorer Micro with 6mm lenses** = wide-FOV tripwire

* ~220° coverage (4 × 55°)
* Max reliable range: ~200m
* Layer 1 is the TRIPWIRE, not the tracker — hands off bearing to radar (Layer 2) for precise 3D localization
* Radar handles the full 300m engagement envelope
