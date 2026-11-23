# Running the learner in a Linux Mint VM

## What must run inside the VM

Install and launch both Steam and Vampire Survivors inside the Linux Mint guest.
The controller captures the guest's desktop and sends input to the game's X11
window; it cannot control a copy of the game running on the Windows host. Use a
Cinnamon **X11** session. The Linux bridge uses `wmctrl`, `xdotool`, and MSS
screen capture, and it deliberately stops with an error on Wayland.

Set the game to fullscreen at a stable resolution and leave its window visible
and focused during training. The VM must stay running and the host must not
suspend it during a long training session.

## Transfer the project and saved progress

Use the VM software's shared-folder feature or another file-transfer method to
copy this project into the guest. Include the Python source files, `data/`, and
the calibration images in `data/`. The DQN checkpoint is
`data/visual_dqn_inlaid_library_8way.pt`; keeping the rest of `data/` also
preserves upgrade preferences and visual templates. `run-evidence/` is optional
and can be left behind to reduce transfer size.

The `.pt` file stores PyTorch model and optimizer state and is separate from the
game's save. Keep a backup of the original data folder before the first Linux
run. Linux training writes new checkpoints into the guest's project folder.

## Install Linux dependencies

From a terminal in the copied project folder:

```bash
chmod +x setup_linux.sh
./setup_linux.sh
```

The setup installs the desktop tools and creates a local Python virtual
environment with OpenCV, NumPy, MSS, and CPU-only PyTorch. Tesseract OCR is
installed by APT and is used for menu and upgrade text; Windows continues to
use the existing PowerShell OCR script. PyTorch's official install selector
lets you choose Linux / Pip / Python / CPU if the VM later gains supported GPU
access: [PyTorch Start Locally](https://pytorch.org/get-started/locally/).

## Start training

1. Log into the Mint Cinnamon X11 desktop.
2. Open Steam and start Vampire Survivors in the VM; leave the title menu visible.
3. In a terminal from the project directory, run:

   ```bash
   source .venv/bin/activate
   python -B -u ./train_agent.py --full-run --deep-learning
   ```

The controller selects Gennaro and Inlaid Library, then repeats runs after
death. Press **Ctrl+C** in that terminal to stop; the checkpoint is saved on
exit and at run end. The first run should be observed to confirm that OCR picks
the intended upgrade cards and that screen clicks align at the VM's resolution.

To check that the Linux dependencies import before starting the game:

```bash
source .venv/bin/activate
python -B -c 'import cv2, mss, torch, train_agent; print("Linux imports OK")'
```

## VM limits

The DQN currently trains on CPU unless CUDA is available in the guest. Most
ordinary VMs do not expose a CUDA GPU, so expect a slower learner than on a
physical NVIDIA GPU. More CPU cores help the OS, OCR, and game, but do not
replace a GPU for convolutional training. Keep the VM window open and test a
short session before leaving it unattended.

