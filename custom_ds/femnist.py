'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2022-03-26 21:09:31
LastEditors: ZhangHongYu
LastEditTime: 2022-03-29 16:25:33
'''
'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2022-02-19 11:25:50
LastEditors: ZhangHongYu
LastEditTime: 2022-03-13 14:41:32
'''
from genericpath import exists
import os

from cvxpy import entr
from utils.split_text import split_into_plays, get_train_test_by_character, write_data_by_character
from sklearn.model_selection import train_test_split
from utils.download import download_url, check_integrity
import re
from typing import Any, Callable, List, Optional
import time
import hashlib
import json
import urllib
import urllib.request
import urllib.error
import torch
import zipfile
import random
from torch.utils.model_zoo import tqdm
try:
    from torchvision.version import __version__ as __vision_version__   # noqa: F401
except ImportError:
    __vision_version__ = "undefined"
import numpy as np
import pickle # 这个包可以读取gz格式的压缩文件
from torch.utils.data import Dataset
import string
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize

USER_AGENT = os.environ.get(
    "TORCHVISION_USER_AGENT",
    f"pytorch-{torch.__version__}/vision-{__vision_version__}"
)


class FEMNIST(Dataset):
    """ FEMNIST <https://leaf.cmu.edu/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``Shakespeare`` exists or will be saved to if download is set to True
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'FEMNIST'
    by_class_url = "https://s3.amazonaws.com/nist-srd/SD19/by_class.zip"
    by_write_url = "https://s3.amazonaws.com/nist-srd/SD19/by_write.zip"
    by_class_filename = "by_class.zip"
    by_write_filename = "by_write.zip"
    by_class_zip_md5 = '79572b1694a8506f2b722c7be54130c4'
    by_write_zip_md5 = 'a29f21babf83db0bb28a2f77b2b456cb'
    
    #txt_filename = "100.txt" #the name of .txt file containing the training corpus
    n_classes = 62

    def __init__(
            self,
            root: str,
            download: bool = False,
            train_frac: float = 0.8,
            val_frac: float = 0,
            transform = None
        ):

        self.root, self.train_frac, self.val_frac, self.transform = root, train_frac, val_frac, transform
        
        # self.train = train  # training set or test set

        self.by_class_zip_path = os.path.join(self.root, self.base_folder, self.by_class_filename)
        self.by_write_zip_path = os.path.join(self.root, self.base_folder, self.by_write_filename)      
        
        if download:
            self.download()
            
        if not check_integrity(self.by_class_zip_path, self.by_class_zip_md5):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')    

        if not check_integrity(self.by_write_zip_path, self.by_write_zip_md5):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')    

        self.generate_dir()


        if not os.path.exists(self.raw_data_dir): 
            self.extract_data()

        if not os.path.exists(self.processed_data_dir):
            self.get_file_dirs()

        if not os.path.exists(os.path.join(self.processed_data_dir, "write_with_class.pkl")):
            self.get_and_match_hashes()
            
        if not os.path.exists(os.path.join(self.processed_data_dir, 'images_by_writer.pkl')):
            self.group_by_writer()

        if not os.path.exists(self.tensor_data_dir):
            self.data_to_tensor()
            

        if not os.path.exists(self.all_data_dir):
            self.generate_data()


        if not os.path.exists(os.path.join(self.processed_data_dir, "dataset.pt")):
            self.preprocess_data()


        self.read_data()  

    def download(self) -> None:

        # print("the zip path is: %s" % self.zip_path)
        if check_integrity(self.by_class_zip_path, self.by_class_zip_md5):
            print('Files already downloaded and verified')
            return
        if check_integrity(self.by_write_zip_path, self.by_write_zip_md5):
            print('Files already downloaded and verified')
            return
        
        download_root = os.path.expanduser(os.path.join(self.root, self.base_folder))
        # if extract_root is None:
        #     extract_root = download_root
        if not self.by_class_filename:
            self.by_class_filename = os.path.basename(self.by_class_url)
        
        if not self.by_write_filename:
            self.by_write_filename = os.path.basename(self.by_write_url)

        download_url(self.by_class_url, download_root, self.by_class_filename, self.by_class_zip_md5)
        download_url(self.by_write_url, download_root, self.by_write_filename, self.by_write_zip_md5)

    def generate_dir(self):
        self.data_dir = os.path.join(self.root, self.base_folder)
        self.by_class_file_path = os.path.join(self.data_dir, self.by_class_filename)
        self.by_write_file_path = os.path.join(self.data_dir, self.by_write_filename)
        self.raw_data_dir = os.path.join(self.data_dir, "raw")
        self.processed_data_dir = os.path.join(self.data_dir, "processed")
        self.tensor_data_dir = os.path.join(self.processed_data_dir, 'data_as_tensor_by_writer')
        self.all_data_dir = os.path.join(self.data_dir, "all_data")
        # if not os.path.exists(os.path.join(self.data_dir, 'processed')):
        #     os.mkdir(os.path.join(self.data_dir, 'processed'))

    def extract_data(self):
        if not os.path.exists(self.raw_data_dir):
            os.makedirs(self.raw_data_dir)
        with zipfile.ZipFile(self.by_class_file_path, 'r') as zin:
            zin.extractall(self.raw_data_dir)
        with zipfile.ZipFile(self.by_write_file_path, 'r') as zin:
            zin.extractall(self.raw_data_dir)


    def get_file_dirs(self):

        def save_obj(obj, name):
            with open(name + '.pkl', 'wb') as f:
                pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

        class_files = []  # (class, file directory)
        write_files = []  # (writer, file directory)

        class_dir = os.path.join(self.raw_data_dir, 'by_class')
        classes = os.listdir(class_dir)
        classes = [c for c in classes if len(c) == 2]

        for cl in classes:
            cldir = os.path.join(class_dir, cl)
            rel_cldir = os.path.join(class_dir, cl)
            subcls = os.listdir(cldir)

            subcls = [s for s in subcls if (('hsf' in s) and ('mit' not in s))]

            for subcl in subcls:
                subcldir = os.path.join(cldir, subcl)
                rel_subcldir = os.path.join(rel_cldir, subcl)
                images = os.listdir(subcldir)
                image_dirs = [os.path.join(rel_subcldir, i) for i in images]

                for image_dir in image_dirs:
                    class_files.append((cl, image_dir))


        write_dir = os.path.join(self.raw_data_dir, 'by_write')
        write_parts = os.listdir(write_dir)
        write_parts = [wp for wp in write_parts if len(wp) == 5]  

        for write_part in write_parts:
            writers_dir = os.path.join(write_dir, write_part)
            rel_writers_dir = os.path.join(write_dir, write_part)
            writers = os.listdir(writers_dir)
            writers = [w for w in writers if len(w) == 8]  

            for writer in writers:
                writer_dir = os.path.join(writers_dir, writer)
                rel_writer_dir = os.path.join(rel_writers_dir, writer)
                wtypes = os.listdir(writer_dir)
                wtypes = [wt for wt in wtypes if len(wt) == 8]

                for wtype in wtypes:
                    type_dir = os.path.join(writer_dir, wtype)
                    rel_type_dir = os.path.join(rel_writer_dir, wtype)
                    images = os.listdir(type_dir)
                    image_dirs = [os.path.join(rel_type_dir, i) for i in images]

                    for image_dir in image_dirs:
                        write_files.append((writer, image_dir))
        
        if not os.path.exists(os.path.join(self.processed_data_dir)):
            os.makedirs(self.processed_data_dir)
            
        save_obj(
            class_files,
            os.path.join(self.processed_data_dir, 'class_file_dirs'))
        save_obj(
            write_files,
            os.path.join(self.processed_data_dir, 'write_file_dirs'))


    def group_by_writer(self):
        def load_obj(name):
            with open(name + '.pkl', 'rb') as f:
                return pickle.load(f)


        def save_obj(obj, name):
            with open(name + '.pkl', 'wb') as f:
                pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


        parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

        wwcd = os.path.join(self.processed_data_dir, 'write_with_class')
        write_class = load_obj(wwcd)

        writers = []  # each entry is a (writer, [list of (file, class)]) tuple
        cimages = []
        (cw, _, _) = write_class[0]
        for (w, f, c) in write_class:
            if w != cw:
                writers.append((cw, cimages))
                cw = w
                cimages = [(f, c)]
            cimages.append((f, c))
        writers.append((cw, cimages))

        ibwd = os.path.join(self.processed_data_dir, 'images_by_writer')
        save_obj(writers, ibwd)

    def get_and_match_hashes(self):
        def load_obj(name):
            with open(name + '.pkl', 'rb') as f:
                return pickle.load(f)


        def save_obj(obj, name):
            with open(name + '.pkl', 'wb') as f:
                pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


        cfd = os.path.join(self.processed_data_dir, 'class_file_dirs')
        wfd = os.path.join(self.processed_data_dir, 'write_file_dirs')

        class_file_dirs = load_obj(cfd)
        write_file_dirs = load_obj(wfd)

        class_file_hashes = []
        write_file_hashes = []

        count = 0
        for tup in class_file_dirs:
            if count % 100000 == 0:
                print('hashed %d class images' % count)

            (cclass, cfile) = tup
            file_path = os.path.join(cfile)

            chash = hashlib.md5(open(file_path, 'rb').read()).hexdigest()

            class_file_hashes.append((cclass, cfile, chash))

            count += 1

        cfhd = os.path.join(self.processed_data_dir, 'class_file_hashes')
        save_obj(class_file_hashes, cfhd)

        count = 0
        for tup in write_file_dirs:
            if count % 100000 == 0:
                print('hashed %d write images' % count)

            (cclass, cfile) = tup
            file_path = os.path.join(cfile)

            chash = hashlib.md5(open(file_path, 'rb').read()).hexdigest()

            write_file_hashes.append((cclass, cfile, chash))

            count += 1

        wfhd = os.path.join(self.processed_data_dir, 'write_file_hashes')
        save_obj(write_file_hashes, wfhd)     

        cfhd = os.path.join(self.processed_data_dir, 'class_file_hashes')
        wfhd = os.path.join(self.processed_data_dir, 'write_file_hashes')
        class_file_hashes = load_obj(cfhd)  # each elem is (class, file dir, hash)
        write_file_hashes = load_obj(wfhd)  # each elem is (writer, file dir, hash)

        class_hash_dict = {}
        for i in range(len(class_file_hashes)):
            (c, f, h) = class_file_hashes[len(class_file_hashes)-i-1]
            class_hash_dict[h] = (c, f)

        write_classes = []
        for tup in write_file_hashes:
            (w, f, h) = tup
            write_classes.append((w, f, class_hash_dict[h][0]))

        wwcd = os.path.join(self.processed_data_dir, 'write_with_class')
        save_obj(write_classes, wwcd)


    def data_to_tensor(self):
        def relabel_class(c):
            """
            maps hexadecimal class value (string) to a decimal number
            returns:
            - 0 through 9 for classes representing respective numbers
            - 10 through 35 for classes representing respective uppercase letters
            - 36 through 61 for classes representing respective lowercase letters
            """
            if c.isdigit() and int(c) < 40:
                return int(c) - 30
            elif int(c, 16) <= 90:  # uppercase
                return int(c, 16) - 55
            else:
                return int(c, 16) - 61


        by_writers_dir = os.path.join(self.processed_data_dir, 'images_by_writer.pkl')
        save_dir = self.tensor_data_dir

        os.makedirs(save_dir, exist_ok=True)

        with open(by_writers_dir, 'rb') as f:
            writers = pickle.load(f)

        for (w, l) in tqdm(writers):

            data = []
            targets = []

            size = 28, 28  # original image size is 128, 128
            for (f, c) in l:
                file_path = os.path.join(f)
                img = Image.open(file_path)
                gray = img.convert('L')
                gray.thumbnail(size, Image.ANTIALIAS)
                arr = np.asarray(gray).copy() / 255  # scale all pixel values to between 0 and 1

                nc = relabel_class(c)

                data.append(arr)
                targets.append(nc)

            if len(data) > 2:
                data = torch.tensor(np.stack(data))
                targets = torch.tensor(np.stack(targets))

                trgt_path = os.path.join(save_dir, w)

                torch.save((data, targets), os.path.join(save_dir, f"{w}.pt"))


    def generate_data(self):
       
        seed = 42
        rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
        rng = random.Random(rng_seed)


        raw_data_path = self.tensor_data_dir

        # n_tasks = int(len(os.listdir(raw_data_path)))
        n_tasks = 1
        file_names_list = os.listdir(raw_data_path)
        rng.shuffle(file_names_list)

        #file_names_list = file_names_list[:n_tasks]

        #os.makedirs(os.path.join(target_path, "train"), exist_ok=True)
        #os.makedirs(os.path.join(target_path, "test"), exist_ok=True)

        target_path = self.all_data_dir  


        n_tasks = len(os.listdir(raw_data_path))
        file_names_list = os.listdir(raw_data_path)
        rng.shuffle(file_names_list)

        file_names_list = file_names_list[:n_tasks]
        rng.shuffle(file_names_list)

        print("generating data..")
        for idx, file_name in enumerate(tqdm(file_names_list)):

            data, targets = torch.load(os.path.join(raw_data_path, file_name))
            train_data, test_data, train_targets, test_targets =\
                train_test_split(
                    data,
                    targets,
                    random_state=seed
                )

            if self.val_frac > 0:
                train_data, val_data, train_targets, val_targets = \
                    train_test_split(
                        train_data,
                        train_targets,
                        train_size=1.-self.val_frac,
                        random_state = seed
                    )

            else:
                val_data, val_targets = None, None

            save_path = os.path.join(target_path, f"task_{idx}")
            os.makedirs(save_path, exist_ok=True)

            torch.save((train_data, train_targets), os.path.join(save_path, "train.pt"))
            torch.save((test_data, test_targets), os.path.join(save_path, "test.pt"))

            if (val_data is not None) and (val_targets is not None):
                torch.save((val_data, val_targets), os.path.join(save_path, "val.pt"))

    def preprocess_data(self):
        self.transform = Compose([
            ToTensor(),
            Normalize((0.1307,), (0.3081,))
        ])
        self.client_train_idcs, self.client_test_idcs, self.client_val_idcs = [], [], []
        start_idcs = 0
        self.data, self.targets = [], []
        dir_path = self.all_data_dir 
        for client_name in os.listdir(self.all_data_dir):

            train_data, train_targets = torch.load(os.path.join(dir_path, client_name, 'train.pt'))

            #with open(os.path.join(dir_path, client_name, 'test.pt'), 'r') as f:
            test_data, test_targets = torch.load(os.path.join(dir_path, client_name, 'test.pt'))

            if self.val_frac > 0:
                valid_data, valid_targets = torch.load(os.path.join(dir_path, client_name, 'val.pt'))
        

            client_n_train, client_n_test = train_data.size(0), test_data.size(0)

            if self.val_frac > 0:
                client_n_val = valid_data.size(0)     

            self.data.append(train_data)
            self.data.append(test_data)
            self.targets.append(train_targets)
            self.targets.append(test_targets)

            if self.val_frac > 0:
                self.data.append(valid_data)
                self.targets.append(valid_targets)


            self.client_train_idcs.append(range(start_idcs, start_idcs + client_n_train))
            start_idcs += client_n_train
            self.client_test_idcs.append(range(start_idcs, start_idcs + client_n_test))      
            start_idcs += client_n_test

            if self.val_frac > 0:
                self.client_val_idcs.append(range(start_idcs, start_idcs + client_n_val))   
                start_idcs += client_n_val

        self.n_sample = start_idcs        

        if not os.path.exists(self.processed_data_dir):
            os.makedirs(self.processed_data_dir)

        entry = {'n_sample': self.n_sample, 'client_train_idcs':self.client_train_idcs, 'client_test_idcs':self.client_test_idcs}
        if self.val_frac > 0:
            entry.update({'client_val_idcs': self.client_val_idcs})
        with open(os.path.join(self.processed_data_dir, 'client_div_info.pt'), 'wb') as f:
            pickle.dump(entry, f)


        self.data, self.targets = np.concatenate(self.data, axis=0), np.concatenate(self.targets, axis=0)
        entry = {'data': self.data, 'targets':self.targets}
       
        with open(os.path.join(self.processed_data_dir, 'dataset.pt'), 'wb') as f:
            pickle.dump(entry, f)    


    def read_data(self):
        # if self.train == True:
        #     mod = "train"
        # else:
        #     mod = "test"
            
        with open(os.path.join(self.processed_data_dir, 'dataset.pt'), 'rb') as f:
            entry = pickle.load(f)
        self.data, self.targets = entry['data'], entry['targets']   
        with open(os.path.join(self.processed_data_dir, 'client_div_info.pt'), 'rb') as f:
            entry = pickle.load(f)  
        self.n_sample, self.client_train_idcs, self.client_test_idcs = entry['n_sample'], entry['client_train_idcs'], entry['client_test_idcs']
        if self.val_frac > 0:
            self.client_val_idcs = entry['client_val_idcs']

    def __len__(self):
        return self.n_sample

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        
        img = np.uint8(np.array(img) * 255)
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target
    
