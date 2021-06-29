import numpy as np
import cv2
import json
import os

# Specify the dataset directories and the classes file path
data_dir = 'm2caiSegdataset'
json_path = 'miccaiSegClasses.json'

# Save directory
save_dir = 'miccaiSegRefined'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


files = {x: [os.path.join(data_dir, x, 'groundtruth', f) for f in os.listdir(os.path.join(data_dir, x, 'groundtruth'))
             if (f.endswith('.jpg') or f.endswith('.png'))]
         for x in ['train', 'test', 'trainval']}



def disentangleKey(key):
    '''
        Disentangles the key for class and labels obtained from the
        JSON file
        Returns a python dictionary of the form:
            {Class Id: RGB Color Code as numpy array}
    '''
    dKey = {}
    for i in range(len(key)):
        class_id = int(key[i]['id'])
        c = key[i]['color']
        c = c.split(',')
        c0 = int(c[0][1:])
        c1 = int(c[1])
        c2 = int(c[2][:-1])
        color_array = np.asarray([c0, c1, c2])
        dKey[class_id] = color_array

    return dKey



# Get the classes RGB key
classes = json.load(open(json_path))['classes']
key = disentangleKey(classes)


# Please run only once, otherwise restart kernel and then run again
for k in range(len(key)):
    rgb = key[k]
    rgb = np.expand_dims(rgb, 0)
    if 'keyMat' in locals():
        keyMat = np.concatenate((keyMat, rgb), axis=0)
    else:
        keyMat = rgb


# Iterate over all images to smooth them
x = ['train', 'test', 'trainval']
for i in range(len(files)):
    folder = files[x[i]]
    save_sub_dir = os.path.join(save_dir, x[i], 'groundtruth')
    if not os.path.exists(save_sub_dir):
        os.makedirs(save_sub_dir)

    for j in range(len(folder)):
        img = cv2.imread(folder[j])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        refined = np.zeros_like(img)

        # Iterate over all image pixels
        # TODO: Vectorize this
        for h in range(img.shape[0]):
            for w in range(img.shape[1]):
                label = np.argmin(np.linalg.norm(np.subtract(img[h, w, :], keyMat), axis=1))
                rgb = key[label]
                refined[h, w, :] = rgb

        # Apply median filtering to remove the salt pepper noise produced at image boundaries
        refined = cv2.medianBlur(refined, 5)

        # Save the image
        refined = cv2.cvtColor(refined, cv2.COLOR_BGR2RGB)
        file_name = folder[j].split('\\')[-1].split('.')[0] + '.png'
        save_path = os.path.join(save_sub_dir, file_name)
        cv2.imwrite(save_path, refined)
        print('Image: [%d]/[%d]: Folder: [%d]/[%d]' % (j + 1, len(folder), i + 1, len(files)))