import sys

sys.path.append('..')
sys.path.append('../models')
from utils import *
from mobnet import ModelMobnet

data_path = '../data_2020/'
val_csv = data_path + 'evaluation_setup/fold1_evaluate.csv'
feat_path = '../features/logmel128_scaled_d_dd/'
experiments = '../train/exp_mobnet/'
model_path = experiments + 'model_name.pth'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_freq_bin = 128
X_val, y_val = load_data_2020(feat_path, val_csv, num_freq_bin, 'logmel')
X_val = np.transpose(X_val,(0,3,1,2)) 

validloader = torch.utils.data.DataLoader([[X_val[i], y_val[i]] for i in range(len(y_val))], 
                                            batch_size=batch_size, num_workers=2) 


net = ModelMobnet()
net.load_state_dict(torch.load(model_path))
net.eval()

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in validloader:
        inputs, labels = data[0].to(device), data[1].type(torch.LongTensor).to(device)
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network: {100 * correct // total} %')