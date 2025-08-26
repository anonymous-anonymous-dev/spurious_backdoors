#!/usr/bin/env bash

conda activate u_torch
conda env export | grep -v "^prefix: " > u_torch.yml
