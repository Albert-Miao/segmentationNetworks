
import numpy as np
import cv2
import json
import os

# Specify the dataset directories and the classes file path
data_dir = 'miccaiSegRefined'
json_path = 'miccaiSegClasses.json'

# Save directory
save_dir = 'miccaiSegOrgans'

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
        category = key[i]['name']
        super_category = key[i]['super-category']
        c = key[i]['color']
        c = c.split(',')
        c0 = int(c[0][1:])
        c1 = int(c[1])
        c2 = int(c[2][:-1])
        color_array = np.asarray([c0, c1, c2])
        dKey[class_id] = {'color': color_array, 'name': category, 'super_category': super_category}

    return dKey



# Get the classes RGB key
classes = json.load(open(json_path))['classes']
key = disentangleKey(classes)

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

        # Iterate over all categories
        for k in range(len(key)):
            rgb = key[k]['color']
            mask = np.all(img == rgb, axis=2)

            # Make all the instruments a single category
            if key[k]['super_category'] == 'instrument':
                img[mask] = np.asarray([0, 85, 170])

            # Make fluids and artery part of the gall-bladder
            if key[k]['super_category'] == 'fluid':
                img[mask] = np.asarray([85, 170, 255])

        # Save the image
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        file_name = folder[j].split('\\')[-1].split('.')[0] + '.png'
        save_path = os.path.join(save_sub_dir, file_name)
        cv2.imwrite(save_path, img)
        print('Image: [%d]/[%d]: Folder: [%d]/[%d]' % (j + 1, len(folder), i + 1, len(files)))
