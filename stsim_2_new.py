from perceptual.metric_copy import Metric
import cv2
import os
import glob
import heapq

data_dir = "/Users/yuxiao/Desktop/data/Corbis128BigExperiment_gray/"
data = glob.glob(data_dir + "*.tiff")
#-------------------------------------
import torch
import time
from torch.utils.data import Dataset, DataLoader

class ImgData(Dataset):

    def __init__(self, k, data):

        self.data = data
        self.img1 = cv2.imread(data[k], cv2.IMREAD_GRAYSCALE)

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        img2_path = self.data[idx]
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        score = m.STSIM2(self.img1, img2)
        
        sample = score
        return sample

def del_path(s):
    (_, temp) = os.path.split(s)
    return temp


def takesecond(elem):
    return elem[1]


m = Metric()
res = []
knum = 10

for k in range(len(data)):
#for k in range(1):
    tmp = []
    score = []
    img1name = del_path(data[k])
    tmp.append(img1name)

    dataset = ImgData(k, data)

    #print(len(dataset))

    dataloader = DataLoader(dataset,
                            batch_size = 16,
                            shuffle = False,
                            num_workers = 16,
                            pin_memory = True)

    score_list = []
    for idx, batch_data in enumerate(dataloader):
        
        score_list.extend(batch_data.numpy().tolist())
        
    max_num_index_list = list(map(score_list.index, heapq.nlargest(knum, score_list)))
    for ind in max_num_index_list:
        tmp.append(del_path(data[ind]))
        score.append(score_list[ind])
    tmp.extend(score)
    res.append(tmp)

    if k%10 == 0:
        print("%d images done"%(k+1))

#-----------------------------------------
outputfile = "./stsim_2_global_result.txt"
with open(outputfile, 'a') as f:
    for i in range(len(res)):
        line = ''
        for name in res[i]:
            line = line + str(name) + ','
        line = line[:-1] + '\n'
        f.write(line)
    f.close()
"""m = Metric()
res = []
knum = 10
#for k in range(len(data)):
for k in range(1):
    img1name = del_path(data[k])
    res.append(img1name)
    img1 = cv2.imread(data[k], cv2.IMREAD_GRAYSCALE)
    stsim2 = []
    for j in range(len(data)):
        img2 = cv2.imread(data[j], cv2.IMREAD_GRAYSCALE)
        score = m.STSIM2(img1, img2)
        stsim2.append([j, score])
    stsim2.sort(key=takesecond, reverse=True)
    for i in range(knum):
        top_name = del_path(data[stsim2[i][0]])
        res.append(top_name)
res = np.array(res).reshape(-1, knum+1)"""


