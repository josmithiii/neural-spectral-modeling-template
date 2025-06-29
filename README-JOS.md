```tcsh
uv venv
uv pip install -r requirements.txt

make train
echo It worked! 31.202u 7.701s 0:33.96 114.5%	0+0k 0+0io 2pf+0w
echo See ./logs/train/runs/2025-06-25_20-57-53/train.log
echo Also interesting: ./logs/train/runs/2025-06-25_20-57-53/config_tree.log

make trainmps
echo Also worked!: 45.044u 4.138s 1:04.19 76.6%	0+0k 0+0io 994pf+0w
echo See ./logs/train/runs/2025-06-25_21-23-45/
```
