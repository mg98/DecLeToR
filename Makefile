# Python interpreter
PYTHON = python

build:
	git submodule update --init --recursive
	make -C allRank install-reqs
	pip install torchvision==0.14.1 torch==1.13.1
	pip install -r requirements.txt

fetch:
	$(PYTHON) fetch.py

run:
	$(PYTHON) main.py

all: fetch run

.PHONY: fetch run all build
