# Vampire Survivors Trainer

A local, vision-based Python trainer for Vampire Survivors. It controls the
visible game window, collects XP gems, avoids threats, chooses upgrades across
the standard level-up menu, a six-card themed choice, and the four-column
Weapon Selection screen, and learns movement with a small visual DQN. Training
progress can be viewed from a LAN dashboard with a live gameplay feed.
Full-run mode selects Poe and Mad Forest, then starts another run after Game Over.

## Requirements

- Vampire Survivors installed through Steam and open in the same desktop session
- Windows 10/11 or Linux Mint / another X11 Linux desktop (Wayland is not
  supported by the Linux input bridge)
- Python 3.10 or newer

## Windows setup

Open a normal, non-elevated PowerShell window in this project folder. Install
the basic controller dependencies with:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_windows.ps1
```

To install PyTorch for the optional visual DQN, run:

```powershell
.\setup_windows.ps1 -WithDeepLearning
```

The script creates a local `.venv`, installs dependencies, and checks whether
PyTorch can use CUDA. Start Steam and Vampire Survivors, leave the game visible
and focused, then start unattended training with:

```powershell
.\.venv\Scripts\python.exe -B -u .\train_agent.py --full-run --deep-learning
```

Without PyTorch, omit `--deep-learning`. Windows uses the GDI/SendInput bridge
and PowerShell OCR. Keep the game visible and use a normal user terminal; the
controller cannot send input to an elevated game from a non-elevated process.

## Linux Mint or other X11 Linux

The Linux setup script installs system tools, Python packages, and CPU PyTorch:

```bash
bash setup_linux.sh
```

Keep the game visible at a stable resolution. Start from the title screen, or
leave a run/upgrade choice open for the trainer to resume. For an unattended
run:

```bash
bash run-local.sh --full-run --deep-learning
```

The Linux bridge uses `xdotool`, `wmctrl`, `mss`, and Tesseract. See
[`LINUX_MINT.md`](LINUX_MINT.md) for VM-specific setup and limitations.

## LAN dashboard

Start the dashboard from the project directory:

```powershell
.\.venv\Scripts\python.exe status_server.py --host 0.0.0.0 --port 8765
```

On Linux, use `.venv/bin/python status_server.py --host 0.0.0.0 --port 8765`.
Open `http://<computer-address>:8765/` from a device on the same network. The
dashboard can launch the game through Steam, start training, stop and save,
request a clean game close, and display live progress. It has no user
authentication; use it only on a trusted private network and never expose it
directly to the public internet.

## Project data and game assets

Model checkpoints, progress reports, run captures, and policy history are
local runtime data and are not included. Template images are also omitted
because they are crops from the game; chest tracking degrades gracefully when
its local template is absent.

This repository does not include a license file.

