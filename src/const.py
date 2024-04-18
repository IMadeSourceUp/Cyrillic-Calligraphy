from pathlib import Path

DIR = Path.cwd() # work directory
PREDICT_PATH = Path(DIR, 'predict')
CHECKPOINTS_PATH = Path(DIR)
FROM_CHECKPOINT_PATH = None # if not None then training start with this checkpoint
WEIGHTS_PATH = Path(DIR, 'ocr_transformer_rn50_4h2l_64x256.pt')
