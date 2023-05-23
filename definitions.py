import os

ROOT_FOLDER = os.path.dirname(os.path.abspath(__file__))
model_save_dir = os.path.join(ROOT_FOLDER, 'models')
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)