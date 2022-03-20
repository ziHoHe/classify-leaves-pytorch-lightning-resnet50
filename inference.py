from train import *
import os
import pandas as pd
import time
from glob import glob

df = pd.read_csv(os.path.join(IMAGE_ROOT_PATH, DATASET_NAME, 'train.csv'))

label_sort_list = sorted(set(list(df.label)))
n_class = len(label_sort_list)
label_map_dict = dict(zip(label_sort_list, range(n_class)))
label_inv_map_dict = {v: k for k, v in label_map_dict.items()}
print(label_inv_map_dict)

best_model = glob('./best_ckpt/checkpoints/*')[0]
pred_df = pd.read_csv(os.path.join(IMAGE_ROOT_PATH, DATASET_NAME, 'test.csv'))

pred_dm = LeaDataModule(batch_size=BATCH_SIZE, test_img_names=pred_df['image'].to_numpy())
pred_model = LeaNetModule.load_from_checkpoint(best_model)

if not os.path.exists('./pred'):
    os.mkdir('./pred')
pred_labels = []

device = 'cuda' if torch.cuda.is_available() else 'cpu'

pred_dm.setup()
dataloader = pred_dm.predict_dataloader()
pred_model.eval()

pred_model.to(device)

label_list = []

for idx, (x, y) in enumerate(dataloader):
    size = len(dataloader.dataset)
    with torch.no_grad():
        start = time.time()
        x = x.to(device)
        with torch.cuda.amp.autocast():
            ret = pred_model(x)
            logits = F.softmax(ret, dim=1)
            labels = logits.argmax(1).cpu().numpy().tolist()
            label_list.extend(labels)

            current = idx * len(x)
            print(f"[{current:>5d}/{size:>5d}]  used time: {(time.time() - start)}")

pred_df['label'] = label_list
pred_df['label'] = pred_df['label'].apply(lambda x: label_inv_map_dict[x])
if os.path.exists('./pred/test.csv'):
    os.remove('./pred/test.csv')
pred_df.to_csv('./pred/test.csv', index=False)
