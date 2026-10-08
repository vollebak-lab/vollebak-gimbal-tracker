# Raspberry Pi 5 bring-up

## Hardware

The Waveshare module uses the General Driver for Robots board (ESP32) and ST3215 serial bus servos. Vendor documentation specifies a 12 V supply, 30 kg-cm peak torque at 12 V, pan up to +/-180 degrees, and tilt roughly -45 to +90 degrees. The reference firmware is more conservative on the lower tilt endpoint, so this repository defaults to -25 to +80 degrees until the assembly is checked.

### Power and first motion

- Use the vendor-rated 12 V supply for the driver board and servos.
- Do not connect/disconnect servo bus cables while powered.
- Keep cables, fingers, payloads, and hard stops out of the sweep.
- Start with the payload removed and low speed/acceleration.
- Treat cutting 12 V servo power as the physical emergency stop.
- The vendor procedure sets tilt servo ID 1 and pan servo ID 2, then records both mechanical centers before assembly is exercised.

### Serial choices

USB is recommended first because it avoids Pi UART overlay details. Connect the board's USB serial connector, then run:

```bash
source .venv/bin/activate
gimbal-tracker doctor -c config/pi.yaml
```

Typical devices are `/dev/ttyUSB0` (CP210x), `/dev/ttyACM0`, `/dev/ttyAMA0` on Pi 5 GPIO UART, or `/dev/serial0` on some configurations. Add the service user to `dialout` if access is denied:

```bash
sudo usermod -aG dialout "$USER"
```

Log out and back in after changing groups.

## Calibration procedure

The gimbal needs a visible indication of where it points: for example, its payload camera preview, a safe alignment pointer, or a mechanical sight. Never use a hazardous laser.

1. Rigidly mount the observing camera and gimbal. Do not move either afterward.
2. Set the actual camera resolution in `config/pi.yaml`.
3. Run `gimbal-tracker calibrate -c config/pi.yaml`.
4. Use W/A/S/D to point the gimbal at a physical landmark.
5. Click that same landmark in the stationary camera view and press C.
6. Repeat for at least six landmarks spread across the useful workspace, including corners and center.
7. Press Enter to fit and save `config/calibration.json`.
8. Keep the reported RMS error. More than a few degrees usually means an alignment mistake, inadequate point spread, or strong depth/parallax variation.

Calibration must be repeated if either mount moves, the resolution/crop changes, or the relevant target depth changes greatly.

## Service installation

After interactive testing succeeds, edit the paths/user in `scripts/vollebak-gimbal.service`, then:

```bash
sudo cp scripts/vollebak-gimbal.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now vollebak-gimbal
journalctl -u vollebak-gimbal -f
```

The supplied service hosts the dashboard on port 8080 across the Pi's local network interfaces. On a trusted network, open `http://<pi-address>:8080` from another device. The starter UI has no authentication; do not expose this port directly to the internet.
