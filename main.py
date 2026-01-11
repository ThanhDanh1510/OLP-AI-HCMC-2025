### START: CÁC KHAI BÁO CHÍNH - KHÔNG THAY ĐỔI ###
SEED = 0  # Số seed (Ban tổ chức sẽ công bố & thay đổi vào lúc chấm)
# Đường dẫn đến thư mục train
# (đúng theo cấu trúc gồm 4 thư mục cho 4 classes của ban tổ chức)
TRAIN_DATA_DIR_PATH = 'data/train'
# Đường dẫn đến thư mục test
TEST_DATA_DIR_PATH = 'data/test'
### END: CÁC KHAI BÁO CHÍNH - KHÔNG THAY ĐỔI ###

### START: CÁC THƯ VIỆN IMPORT ###
# Lưu ý: các thư viện & phiên bản cài đặt vui lòng để trong requirements.txt
import os
import torch
import numpy as np
import random

import torch.nn as nn
import torch.optim as optim

### END: CÁC THƯ VIỆN IMPORT ###

### START: SEEDING EVERYTHING - KHÔNG THAY ĐỔI ###
# Seeding nhằm đảm bảo kết quả sẽ cố định
# và không ngẫu nhiên ở các lần chạy khác nhau
# Set seed for random
random.seed(SEED)
# Set seed for numpy
np.random.seed(SEED)
# Set seed for torch
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
### END: SEEDING EVERYTHING - KHÔNG THAY ĐỔI ###

# START: IMPORT CÁC THƯ VIỆN CUSTOM, MODEL, v.v. riêng của nhóm ###
from libs.config import get_config
from libs.dataset import get_dataloader
from libs.model import SimpleCNN, init_params
from libs.train import train_model
from libs.predict import predict_test

### END: IMPORT CÁC THƯ VIỆN CUSTOM, MODEL, v.v. riêng của nhóm ###


### START: ĐỊNH NGHĨA & CHẠY HUẤN LUYỆN MÔ HÌNH ###
# Model sẽ được train bằng cac ảnh ở [TRAIN_DATA_DIR_PATH]
train_dataloader, val_dataloader = get_dataloader(
    TRAIN_DATA_DIR_PATH, batch_size=64, num_workers=0, is_valid=False
)

model = SimpleCNN()
init_params(model)

config = get_config()

batch_size = config['BATCH_SIZE']
num_workers = config['NUM_WORKERS']
n_epochs = config['N_EPOCHS']
learning_rate = config['LEARNING_RATE']

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
global_step = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_losses = []
val_losses, val_accuracy = [], []
model, train_losses, val_losses, val_accuracy = train_model(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    optimizer=optimizer,
    criterion=criterion,
    config=config,
    n_epochs=n_epochs,
    n_train=len(train_dataloader)*batch_size
)

### END: ĐỊNH NGHĨA & CHẠY HUẤN LUYỆN MÔ HÌNH ###


### START: THỰC NGHIỆM & XUẤT FILE KẾT QUẢ RA CSV ###
# Kết quả dự đoán của mô hình cho tập dữ liệu các ảnh ở [TEST_DATA_DIR_PATH]
# sẽ lưu vào file "output/results.csv"
# Cấu trúc gồm 2 cột: image_name và label: (encoded: 0, 1, 2, 3)
# image_name,label
# image1.jpg,0
# image2.jpg,1
# image3.jpg,2
# image4.jpg,3
submission = predict_test(model, config)
if not os.path.exists(config['RESULTS_DIR_PATH']):
    os.mkdir(config['RESULTS_DIR_PATH'])
submission.to_csv(config['RESULTS_DIR_PATH'] + '/results.csv', index=False)

### END: THỰC NGHIỆM & XUẤT FILE KẾT QUẢ RA CSV ###
