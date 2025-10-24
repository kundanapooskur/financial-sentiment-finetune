.PHONY: prep train eval baseline hpo demo env

env:
	python -c "import torch,platform;print('Python:',platform.python_version());print('CUDA available:', __import__('torch').cuda.is_available())"

prep:
	python -m src.data_prep --subset --n_train 3000 --n_val 500 --n_test 500

baseline:
	python -m src.eval --baseline

train:
	python -m src.train --epochs 3 --lr 2e-5 --batch 16 --eval-steps 200 --tb

eval:
	python -m src.eval

hpo:
	python -m src.hpo --grid

demo:
	python -m app.demo_gradio
