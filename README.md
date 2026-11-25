# Vampire Survivors Trainer

A local, vision-based Python trainer for Vampire Survivors. It controls the visible game window, collects XP gems, avoids threats, chooses upgrades across the standard level-up menu, a six-card themed choice, and the four-column Weapon Selection screen, and learns movement with a small visual DQN. Training progress can be viewed from a LAN dashboard.

## Requirements

- Vampire Survivors installed through Steam and open in the same desktop session as the trainer
- Linux Mint or another X11 Linux desktop (Wayland is not supported by the Linux input bridge)
- Python 3.10 or newer

The Linux setup script installs system tools, Python packages, and CPU PyTorch:

```bash
bash setup_linux.sh
```

Keep the game visible at a stable resolution. Start from the title screen, or leave a run/upgrade choice open for the trainer to resume. For an unattended run:

```bash
bash run-local.sh --full-run --deep-learning
```

The trainer saves its DQN checkpoint and policy data under `data/`. Those runtime files are ignored by Git. The bundled chest sprite template is optional; without it, the trainer still handles chest reward screens but does not track unopened chests by sprite.

## LAN dashboard

Start the dashboard from the project directory:

```bash
.venv/bin/python status_server.py --host 0.0.0.0 --port 8765
```

Open `http://<vm-address>:8765/` from a device on the same network. The dashboard can launch the game through Steam, start training, stop and save, request a clean game close, and display live progress. It has no user authentication; use it only on a trusted private network and never expose it directly to the public internet.

**Run learning check** uses an existing checkpoint for 10 evaluation runs, 20 training runs, and 10 follow-up evaluation runs. Start from the title or Game Over screen. Both evaluations freeze learning and use the same upgrade policy. The dashboard reports the median and range of in-game survival times, plus the number of timers it could read. Checkpoints, phase reports, and evidence are saved under `data/learning-checks/`.

## Platform notes

The Linux bridge uses `xdotool`, `wmctrl`, `mss`, and Tesseract. Windows support uses the separate Windows bridge and OCR script; install the Python packages listed in `requirements-linux.txt` plus PyTorch in that environment. The Linux setup script is the maintained setup path.

## Project data and game assets

Model checkpoints, progress reports, run captures, and policy history are local runtime data and are not included. Template images are also omitted because they are crops from the game; chest tracking degrades gracefully when its local template is absent.

This repository does not include a license file.
