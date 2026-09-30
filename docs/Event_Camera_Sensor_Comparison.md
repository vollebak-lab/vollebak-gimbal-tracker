# Event Camera Comparison: GenX320 vs DVXplorer Micro

## For Project Predator — Drone Propeller Detection at Range

## Key Specifications

| Spec | **GenX320** | **DVXplorer Micro** |
| -- | -- | -- |
| Resolution | 320×320 | 640×480 |
| Pixel Pitch | 6.3 µm | 9 µm |
| Dynamic Range | 140 dB | ~90–110 dB |
| Power | 3 mW | <700 mW |
| Weight | ~5g | 16g |
| IMU | None | 6-axis onboard |
| SDK | OpenMV MicroPython | dv_processing (C++/Python) |
| Price (×4) | ~$1,680 | ~€10,000 |

## Detection Range (6mm lens)

| Sensor | Max detection (25cm prop) |
| -- | -- |
| GenX320 | ~80m |
| DVXplorer Micro | ~55m |

## Detection Range (16mm lens)

| Sensor | Max detection (25cm prop) |
| -- | -- |
| GenX320 | ~213m |
| DVXplorer Micro | ~148m |

## Verdict

**DVXplorer Micro selected** for Predator:

1. Direct USB to Orin NX (no bridge board)
2. overlab-kevin repo works natively
3. Onboard 6-axis IMU for head motion compensation
4. VGA resolution with 3× FOV coverage
5. 16g per unit (64g total for 4-camera array)
