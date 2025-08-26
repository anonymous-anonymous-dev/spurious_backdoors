#!/usr/bin/env bash

conda create -n u_torch python=3.12
conda activate u_torch
conda env update --name u_torch --file u_torch.yml
conda deactivate
conda activate u_torch
