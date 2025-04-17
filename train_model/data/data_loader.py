import pathlib
import tensorflow as tf

class DataLoader:
    def __init__(self, data_dir='training_data'):
        self.data_dir = pathlib.Path(data_dir)
        self.cap_file = self.data_dir / 'pseudo_caption' / 'pseudo_caption.txt'
        self.img_dir = self.data_dir / 'dataset'

    def load_caps(self):
        caps = {}
        with open(self.cap_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',', 1)
                if len(parts) >= 2:
                    img, cap = parts
                    key = img.split('#')[0]
                    caps.setdefault(key, []).append(cap)
        return caps

    def create_ds(self, caps):
        paths, cap_list = [], []
        for key, cap in caps.items():
            if len(cap) == 5:
                paths.append(str(self.img_dir / key))
                cap_list.append(cap)
        return tf.data.Dataset.from_tensor_slices((paths, cap_list))