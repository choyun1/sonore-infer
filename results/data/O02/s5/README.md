Runs on Cho's PC (Windows 11, Ryzen 5 7600X3D, 6 threads; RTX 2070 SUPER, driver 617.42),
torch 2.14.1+cu130, 2026-10-18, branch claude/project-thread-z3btfl at 79dad5c, run one at a time
with PYTHONPATH=src (the venv's editable install pointed at an older checkout). Commands:

    python tools/online/online_cost.py --device cuda > pc_cost_cuda.txt
    python tools/online/online_cost.py --device cpu > pc_cost_cpu.txt
    python tools/online/online_buildup.py --semitones 8 --lag 1 --steps 2000 --window-patience 50 --device cuda
