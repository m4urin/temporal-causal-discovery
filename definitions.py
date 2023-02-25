import os
import torch

""" Get the full paths"""
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
SETTINGS_DIR = os.path.join(PROJECT_ROOT, 'settings')

""" pytorch """
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

""" GPULab """
GPULAB_JOB_ID = None
if 'GPULAB_JOB_ID' in os.environ:
    GPULAB_JOB_ID = os.environ['GPULAB_JOB_ID'][:6]
    RESULTS_DIR = os.path.join(os.path.split(PROJECT_ROOT)[0], 'job_results', GPULAB_JOB_ID)

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

if __name__ == '__main__':
    print(f'PROJECT_ROOT: {PROJECT_ROOT}')
    print(f'DATA_DIR: {DATA_DIR}')
    print(f'RESULTS_DIR: {RESULTS_DIR}')
