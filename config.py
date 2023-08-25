import os
from datetime import datetime

""" Get the full paths"""
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

""" GPULab """
GPULAB_JOB_ID = None
if 'GPULAB_JOB_ID' in os.environ:
    GPULAB_JOB_ID = os.environ['GPULAB_JOB_ID'][:6]
    RESULTS_DIR = os.path.join(os.path.split(PROJECT_ROOT)[0], 'outputs')

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR, exist_ok=True)


if __name__ == '__main__':
    print(f'PROJECT_ROOT: {PROJECT_ROOT}')
    print(f'DATA_DIR: {DATA_DIR}')
    print(f'RESULTS_DIR: {RESULTS_DIR}')
