.PHONY: help install smoke teacher student distill prune benchmark export all tf clean

help:
	@echo "Targets: install | smoke | all | benchmark | export | tf | clean"
	@echo "  make all       full PyTorch pipeline (teacher->distill->prune->benchmark->export)"
	@echo "  make smoke     fast end-to-end check on a tiny subset"
	@echo "  make tf        TensorFlow -> TFLite variant (train + export + benchmark)"

install:
	pip install -r requirements.txt

smoke:
	python tests/smoke_test.py

teacher:
	python main.py teacher

student:
	python main.py student

distill:
	python main.py distill

prune:
	python main.py prune --amount 0.5

benchmark:
	python main.py benchmark

export:
	python main.py export --ckpt checkpoints/student_pruned.pt --int8

all:
	python main.py all

tf:
	python tf/train_tf.py
	python tf/export_tflite.py

clean:
	rm -rf checkpoints exports results __pycache__ src/__pycache__ tests/__pycache__ tf/artifacts
