import glob
import cv2
import os
from perceptual.metric import Metric

data_dir = "/Users/yuxiao/Desktop/data/Corbis128BigExperiment_gray/"
data = glob.glob(data_dir + "*.tiff")

def del_path(s):
    (filepath, temp) = os.path.split(s)
    #(filename, extension) = os.path.splitext(temp)
    return temp

gidlist = []
gvec_matrix = []
m = Metric()
for i in range(len(data)):
    img = cv2.imread(data[i], cv2.IMREAD_GRAYSCALE)
    #img = np.array(img)
    #vector = img.reshape(1,-1)
    vector = m.STSIM_M(img)
    gidlist.append(del_path(data[i]))
    gvec_matrix.append(vector)


outputfile = "./stsim_m_fea_2.txt"
with open(outputfile, 'w+') as f:
    for k in range(len(gidlist)):
        line1 = str(gidlist[k]) + ' '
        line2 = ''
        for num in gvec_matrix[k]:
            line2 = line2 + str(num) + ' '
        line = line1 + line2[:-1] + '\n'
        f.write(line)
    f.close()
