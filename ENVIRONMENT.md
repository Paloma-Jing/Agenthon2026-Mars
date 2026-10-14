# Development environment

Use Python 3.13. Clone the official Track 2 repository at the commit specified in the root README, create `.venv-agent`, and install `requirements-dev.txt`.

`environment.yml` offers a Conda base environment named `agenthon_mars`. Install `requirements-numeric-agent.txt` after creating it. The lock files record the original resolved dependencies; they are not a portable operating-system lock.

`python verify_environment.py` runs three public baseline tasks with stage and heartbeat logs under `runs/environment_check`. `python tools/smoke_staged_unit.py` checks this project's actual numerical Agent against the official gates. Neither check establishes prediction accuracy.

A Linux/amd64 Docker image is the competition artifact. The local House stub checks protocol shape; it does not call the real House service.
