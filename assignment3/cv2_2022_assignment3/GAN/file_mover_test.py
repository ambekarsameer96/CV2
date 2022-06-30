import os 
import shutil 
from tqdm import tqdm 
fol1 = '/home/lcur1547/cv2/dataset/data_set/data'
print('Folder is ', fol1)

with open('/home/lcur1547/cv2/dataset/data_set/test.str') as f:
    lines = f.readlines()
fol2 = '/home/lcur1547/cv2/dataset/data_set/test_only'
if not(os.path.exists(fol2)):
    os.makedirs(fol2)


for line in tqdm(lines):
    line = line.split('\n')[0]
    f1 = os.path.join(fol1, line)
    print(f1)
    f2 = os.path.join(fol2, line)
    shutil.copy2(f1, f2)
    