export PYTHONPATH=.
accelerate launch experts/generate_depth.py
accelerate launch experts/generate_edge.py
accelerate launch experts/generate_normal.py
accelerate launch experts/generate_objdet.py
accelerate launch experts/generate_ocrdet.py
accelerate launch experts/generate_segmentation.py