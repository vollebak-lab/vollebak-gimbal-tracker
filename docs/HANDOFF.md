# System handoff and daily operation

This package turns the current Windows-laptop + Raspberry Pi deployment into a repeatable launcher. It targets the hardware that has been verified in this repository:

- Windows laptop with the Logitech MX Brio and IDS IMX636 connected by USB
- Ubuntu under WSL 2 with the verified OpenEB CPU detector build
- Raspberry Pi 5 at `192.168.0.3`, connected over direct Ethernet
- Pi login `vollebak`
- Waveshare driver board on Pi device `/dev/ttyUSB0`
- Tilt servo ID `1`; pan servo ID `2`

Tracking always starts **paused**. Starting the software cannot immediately move the gimbal unless the operator explicitly presses **START TRACKING** in the dashboard.

## Daily operation

1. Place the gimbal where it has clearance to move. Connect its external 12 V supply.
2. Connect the Pi Ethernet cable, Logitech camera, and IDS event camera to the laptop.
3. Power on the Raspberry Pi and wait about 30 seconds.
4. Double-click `RUN_SYSTEM.cmd`.
5. Wait for the browser to open at `http://192.168.0.3:8080`.
6. Confirm both camera indicators are online.
7. Keep people and cables clear, then press **START TRACKING**.

At the end of a session, double-click `STOP_SYSTEM.cmd`. It pauses tracking, returns the gimbal home, and stops laptop camera helpers and tunnels. The Pi dashboard remains available; if the optional boot service was installed by `SETUP_HANDOFF.cmd`, it also restarts automatically after a Pi reboot.

## What the launcher does

`RUN_SYSTEM.cmd` calls `scripts/start_system.ps1`, which:

1. Verifies passwordless SSH access to the Pi.
2. Selects the Windows DirectShow device named `MX Brio` (independent of USB index) and starts or verifies its MJPEG bridge on laptop port `8082`.
3. Keeps Ubuntu WSL running.
4. Finds USB VID/PID `1409:8e00` and attaches the IMX636 through `usbipd`.
5. Starts or verifies Bart's OpenEB CPU detector on laptop/WSL port `8081`.
6. Creates reverse SSH tunnels so the Pi sees both laptop services on its own loopback interface.
7. Starts the Pi dashboard if it is not already running.
8. pauses tracking and commands the gimbal to its home position.
9. Prints live health and opens the dashboard.

If the event camera is absent, the launcher warns but continues with Logitech tracking. A failure of the Logitech bridge, Pi SSH connection, or dashboard is treated as a startup failure.

### Low-latency dashboard path

The Logitech panel relays the bridge's original MJPEG bytes directly to the browser in small immediately available chunks. Detection and gimbal control still run on the Pi, but they no longer block or re-encode the visible feed. The target marker is rendered as a browser overlay from 10 Hz telemetry. This avoids the former laptop encode -> Pi decode -> detection -> Pi encode round trip and prevents multi-frame HTTP buffering.

## One-time provisioning

The current development laptop and Pi can already use `RUN_SYSTEM.cmd`. When preparing a fresh checkout or transferring ownership, double-click `SETUP_HANDOFF.cmd` and enter the Pi password when requested. It:

- creates and installs the Windows Python environment when missing;
- creates a dedicated SSH key under `.pi-ssh/`;
- installs that key on the Pi;
- requests Windows administrator approval to share a connected IMX636 when needed;
- deploys `config/pi.handoff.yaml` as the live Pi config;
- installs and enables the `vollebak-gimbal.service` boot service.

The service installation uses `sudo` on the Pi and therefore prompts for the Pi password once.

## One-time setup on a different Windows laptop

A different laptop requires these prerequisites before running `SETUP_HANDOFF.cmd`:

1. Windows OpenSSH client.
2. WSL 2 with an Ubuntu distribution named `Ubuntu`.
3. `usbipd-win` available as `usbipd.exe`.
4. Python 3.11 or newer with the `py.exe` launcher. `SETUP_HANDOFF.cmd` creates the repository virtual environment and installs its Windows camera dependencies automatically.

5. The Ethernet adapter connected to the Pi configured as `192.168.0.2/24`. The Pi is `192.168.0.3/24`.
6. Bart's verified OpenEB installation at `$HOME/metavision-5.2-ids` inside WSL.
7. The CPU detector executable at `$HOME/predator-event-build/ev_flicker_detector` inside WSL.

The OpenEB/IMX636 runtime is the only non-trivial machine-specific dependency. It contains compiled camera plugins and cannot be replaced by copying a Python package. If it is not installed, the launcher still supports Logitech tracking and reports the event camera offline.

If the IMX636 is new to that laptop, open an Administrator PowerShell once and share it:

```powershell
usbipd list
usbipd bind --busid <BUSID shown for 1409:8e00>
```

The daily launcher attaches it to WSL automatically after that one-time share.

## Configuration

The deployment template is `config/pi.handoff.yaml`. Its current profile uses:

- maximum Waveshare speed and acceleration (`speed: 0`, `acceleration: 0`);
- 100 Hz command ceiling;
- 50 Hz event telemetry polling;
- zero smoothing and deadband;
- safe verified limits of pan `-30°..+30°` and tilt `-10°..+15°`;
- tracking disabled at application startup.

The full hardware angle range is intentionally not enabled because cables and mounted payloads must be checked before expanding it.

## Command-line options

Normal safe startup:

```powershell
.\scripts\start_system.ps1
```

Start without opening a browser:

```powershell
.\scripts\start_system.ps1 -NoBrowser
```

Explicitly start with tracking active, only when the area is already clear:

```powershell
.\scripts\start_system.ps1 -StartTracking
```

Stop and detach the IMX636 from WSL:

```powershell
.\scripts\stop_system.ps1 -DetachEventCamera
```

## Troubleshooting

- **Pi unavailable:** verify the direct Ethernet adapter is up and has `192.168.0.2/24`, then ping `192.168.0.3`.
- **Logitech says demo:** rerun `RUN_SYSTEM.cmd`; it recreates the port-8082 tunnel.
- **Wrong/slow RGB camera:** rerun `RUN_SYSTEM.cmd`; it validates the friendly name and rebuilds the bridge on `MX Brio` instead of relying on a USB index.
- **Event tab disabled:** ensure the IMX636 appears in `usbipd list`, then rerun the launcher.
- **Event camera times out:** unplug/reconnect the IMX636 or run `usbipd detach --busid <BUSID>` before rerunning the launcher.
- **Gimbal unavailable:** verify external 12 V power, `/dev/ttyUSB0`, and membership of the Pi user in the `dialout` group.
- **Emergency stop:** use the physical power switch/disconnect for the servo supply. Software is not a substitute for a physical stop.
