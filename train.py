import sys
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from training.train import run_training_pipeline

if __name__ == '__main__':
    run_training_pipeline()
