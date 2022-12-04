import numpy as np
import os
import shutil



# train (80%), validation (15%) , test(5%)
angry = (108,20,7)
contempt = (43 ,8, 3)
disgust = (142,27, 9)
fear = (60 ,11, 4)
happy = (166,31,10)
sad = (67 ,13, 4)
surprise = (199,37,12)

split_values = [(108,20,7), (43 ,8, 3), (142,27, 9), (60 ,11, 4), (166,31,10), (67 ,13, 4), (199,37,12)]
emotions = ['angry','contempt','disgust','fear','happy','sad','surprise']


base_path = 'F:/ia930/2022.2/GiMMP/Codes/ckplus/ck/CK+48/'


def split_ds(tuple_, path_):
    files = os.listdir(path_)
    np.random.shuffle(files)
    train_val = tuple_[0]
    validation_val = tuple_[1]
    test_val = tuple_[2]

    train_files = files[: train_val]
    validation_files = files[train_val : (train_val + validation_val)]
    test_files = files[train_val + validation_val :]
    print(test_files)

    return train_files, validation_files, test_files

def copy_files(src_list, split_type, emotion):
    for filename in src_list:
        shutil.copyfile(base_path + emotion + '/' + filename, base_path + split_type + '/' + emotion + '/' + filename)


for i in range(len(emotions)):
    train, validation, test = split_ds(split_values[i], emotions[i])
    copy_files(train, 'train', emotions[i])
    copy_files(validation, 'validation', emotions[i])
    copy_files(test, 'test', emotions[i])




