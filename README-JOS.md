```tcsh
uv venv
uv pip install -r requirements.txt
make train
echo It worked! See ./logs/train/runs/2025-06-25_20-57-53/train.log
echo Also interesting: ./logs/train/runs/2025-06-25_20-57-53/config_tree.log
```
