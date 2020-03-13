from perceptual.metric import Metric
import cv2
import os
import glob
import heapq
import time
from torch.utils.data import Dataset, DataLoader

data_dir = "/Users/yuxiao/Desktop/data/Corbis128BigExperiment_gray/"
data = glob.glob(data_dir + "*.tiff")


class ImgData(Dataset):

    def __init__(self, k, data):

        self.data = data
        self.img1 = cv2.imread(data[k], cv2.IMREAD_GRAYSCALE)

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        img2_path = self.data[idx]
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        score = m.STSIM(self.img1, img2)
        
        sample = score
        return sample

    
def del_path(s):
    (_, temp) = os.path.split(s)
    return temp


m = Metric()
res = []
knum = 10

t1 = time.time()
for k in range(len(data)):
    tmp = []
    score = []
    img1name = del_path(data[k])
    tmp.append(img1name)

    dataset = ImgData(k, data)
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

    if k and k%255 == 0:
        print("%d images done"%(k+1))
t2 = time.time()
print("time cost: %f s"%(t2-t1))
print(res)

outputfile = "./stsim_1_result.txt"
with open(outputfile, 'w+') as f:
    for i in range(len(res)):
        line = ''
        for name in res[i]:
            line = line + str(name) + ','
        line = line[:-1] + '\n'
        f.write(line)
    f.close()

