import os

import mlflow

""" Get the full paths"""
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')

""" GPULab """
GPULAB_JOB_ID = None
if 'GPULAB_JOB_ID' in os.environ:
    GPULAB_JOB_ID = os.environ['GPULAB_JOB_ID'][:6]
    OUTPUT_DIR = os.path.join(os.path.split(PROJECT_ROOT)[0], 'outputs')

TEST_DIR = os.path.join(OUTPUT_DIR, 'test')
os.makedirs(TEST_DIR, exist_ok=True)

""" MLFlow """
TRACKING_URI = f'file://{os.path.join(OUTPUT_DIR, "mlruns")}'.replace("\\", "/")
mlflow.set_tracking_uri(TRACKING_URI)

if __name__ == '__main__':
    print(f'PROJECT_ROOT: {PROJECT_ROOT}')
    print(f'DATA_DIR: {DATA_DIR}')
    print(f'OUTPUT_DIR: {OUTPUT_DIR}')
    print(f'TEST_DIR: {TEST_DIR}')
