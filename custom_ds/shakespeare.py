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
from utils.split_text import split_into_plays, get_train_test_by_character, write_data_by_character, train_test_split
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


USER_AGENT = os.environ.get(
    "TORCHVISION_USER_AGENT",
    f"pytorch-{torch.__version__}/vision-{__vision_version__}"
)


class Shakespeare(Dataset):
    """ Shakespeare <https://leaf.cmu.edu/>`_ Dataset.
    It's the Dataset for next character prediction, each sample represents an input sequence of characters
        and a target sequence of characters representing to next sequence of the input
    Args:
        root (string): Root directory of dataset where directory
            ``Shakespeare`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'Shakespeare'
    url = "http://www.gutenberg.org/files/100/old/1994-01-100.zip"
    filename = "1994-01-100.zip"
    zip_md5 = 'b8d60664a90939fa7b5d9f4dd064a1d5'
    txt_filename = "100.txt" #the name of .txt file containing the training corpus

    all_characters = string.printable
    vocab_size = len(all_characters)
    n_characters = len(all_characters)
    input_size = len(string.printable)
    embed_size = 8
    hidden_size = 256
    output_size = len(string.printable)
    n_layers = 2
    chunk_len = 80 # the length of the input and target sequences 
    

    def __init__(
            self,
            root: str,
            download: bool = False,
            train_frac: float = 0.8,
            val_frac: float = 0
        ):

        self.root, self.train_frac, self.val_frac = root, train_frac, val_frac
        
        # self.train = train  # training set or test set

        self.zip_path = os.path.join(self.root, self.base_folder, self.filename)
        if download:
            self.download()
            
        if not check_integrity(self.zip_path, self.zip_md5):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')    

        self.generate_dir()

        if not os.path.exists(os.path.join(self.data_dir, 'raw/100.txt')): 
            self.extract_data()

        if not os.path.exists(os.path.join(self.data_dir, 'raw/by_play_and_character')): 
            self.split_data()

        if not os.path.exists(os.path.join(self.data_dir, 'all_data')):
            self.generate_data()

        if not os.path.exists(os.path.join(self.data_dir, 'processed', 'dataset.pt')):
            self.preprocess_data()

        self.read_data()  

    def download(self) -> None:

        # print("the zip path is: %s" % self.zip_path)
        if check_integrity(self.zip_path, self.zip_md5):
            print('Files already downloaded and verified')
            return
        download_root = os.path.expanduser(os.path.join(self.root, self.base_folder))
        # if extract_root is None:
        #     extract_root = download_root
        if not self.filename:
            self.filename = os.path.basename(self.url)

        download_url(self.url, download_root, self.filename, self.zip_md5)

    def generate_dir(self):
        self.data_dir = os.path.join(self.root, self.base_folder)
        self.file_path = os.path.join(self.data_dir, self.filename)
        if not os.path.exists(os.path.join(self.data_dir, 'raw')):
            os.mkdir(os.path.join(self.data_dir, 'raw'))
        # if not os.path.exists(os.path.join(self.data_dir, 'processed')):
        #     os.mkdir(os.path.join(self.data_dir, 'processed'))

    def extract_data(self):
        with zipfile.ZipFile(self.file_path, 'r') as zin:
            zin.extractall(os.path.join(self.data_dir, "raw"))

    def split_data(self):
        """preprocess the shakespeare dataset

        Args:
            file_root: the root path of the shakespeare text file(.txt)
        """
        print('Splitting .txt data between users')
        file_path = os.path.join(self.data_dir, "raw", self.txt_filename)
        if not os.path.exists(file_path):
            raise IOError("Cant find the %s !" % file_path)
        with open(file_path, 'r') as input_file:
            shakespeare_full = input_file.read()
        plays, discarded_lines = split_into_plays(shakespeare_full)
        print('Discarded %d lines' % len(discarded_lines))
        users_and_plays, all_examples, _ = get_train_test_by_character(plays, test_fraction=-1.0)
        output_dir = os.path.join(self.root, self.base_folder, "raw")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(os.path.join(output_dir, 'users_and_plays.json'), 'w') as ouf:
            json.dump(users_and_plays, ouf)

        write_data_by_character(all_examples,
                                os.path.join(output_dir,
                                            'by_play_and_character/'))

    def generate_data(self):
       
        seed = 42
        rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
        rng = random.Random(rng_seed)


        raw_data_path = os.path.join(self.data_dir,  "raw/by_play_and_character")

        # n_tasks = int(len(os.listdir(raw_data_path)))
        n_tasks = 1
        file_names_list = os.listdir(raw_data_path)
        rng.shuffle(file_names_list)

        #file_names_list = file_names_list[:n_tasks]
        rng.shuffle(file_names_list)

        #os.makedirs(os.path.join(target_path, "train"), exist_ok=True)
        #os.makedirs(os.path.join(target_path, "test"), exist_ok=True)

        dir_path = os.path.join(self.data_dir,  "all_data")  

        for idx, file_name in enumerate(file_names_list):
            # if idx < int(n_tasks):
            #     mode = "train"
            # else:
            #     mode = "test"

            if not os.path.exists(os.path.join(dir_path, 'client %d' % idx)):
                os.makedirs(os.path.join(dir_path, 'client %d' % idx))

            client_dir = os.path.join(dir_path, "client %d" % idx)       

            text_path = os.path.join(raw_data_path, file_name)

            with open(text_path, "r") as f:
                raw_text = f.read()

            raw_text = re.sub(r"   *", r' ', raw_text)

            train_text, test_text = train_test_split(raw_text, self.train_frac)

            if self.val_frac > 0:
                train_text, val_text = train_test_split(train_text, 1.- self.val_frac)
                val_text = val_text.replace('\n', ' ')

            else:
                val_text = None

            train_text = train_text.replace('\n', ' ')
            test_text = test_text.replace('\n', ' ')

            with open(os.path.join(client_dir, "train.txt"), 'w') as f:
                f.write(train_text)

            with open(os.path.join(client_dir, "test.txt"), 'w') as f:
                f.write(test_text)

            if val_text is not None:
                with open(os.path.join(client_dir, "val.txt"), 'w') as f:
                    f.write(val_text)

    def preprocess_data(self):

        dir_path = os.path.join(self.data_dir,  "all_data")  
        out_dir_path = os.path.join(self.data_dir,  "processed")  

        self.client_train_idcs, self.client_test_idcs, self.client_val_idcs = [], [], []
        start_idcs = 0
        self.data, self.targets = [], []
        self.__build_mapping()

        for client_name in os.listdir(dir_path):

            with open(os.path.join(dir_path, client_name, 'train.txt'), 'r') as f:
                client_train_text = f.read()

            with open(os.path.join(dir_path, client_name, 'test.txt'), 'r') as f:
                client_test_text = f.read()

            if self.val_frac > 0:
                with open(os.path.join(dir_path, client_name, 'val.txt'), 'r') as f:
                    client_val_text = f.read()


            client_n_train = max(0, len(client_train_text) - self.chunk_len)
            client_n_test = max(0, len(client_test_text) - self.chunk_len)
            client_train_tokenized  = self.__tokenize(client_train_text)
            client_test_tokenized = self.__tokenize(client_test_text)
            client_train_data, client_train_targets = self.__preprocess_text(client_n_train, client_train_tokenized)
            client_test_data, client_test_targets = self.__preprocess_text(client_n_test, client_test_tokenized)

            if self.val_frac > 0:
                client_n_val = max(0, len(client_val_text) - self.chunk_len)             
                client_val_tokenized  = self.__tokenize(client_val_text)
                client_val_data, client_val_targets = self.__preprocess_text(client_n_val, client_val_tokenized)
        

            self.data.append(client_train_data)
            self.data.append(client_test_data)
            self.targets.append(client_train_targets)
            self.targets.append(client_test_targets)

            if self.val_frac > 0:
                self.data.append(client_val_data)
                self.targets.append(client_val_targets)


            self.client_train_idcs.append(range(start_idcs, start_idcs + client_n_train))
            start_idcs += client_n_train
            self.client_test_idcs.append(range(start_idcs, start_idcs + client_n_test))      
            start_idcs += client_n_test
   
            if self.val_frac > 0:
                self.client_val_idcs.append(range(start_idcs, start_idcs + client_n_val))   
                start_idcs += client_n_val

        self.n_sample = start_idcs        

        if not os.path.exists(out_dir_path):
            os.makedirs(out_dir_path)

        entry = {'n_sample': self.n_sample, 'client_train_idcs':self.client_train_idcs, 'client_test_idcs':self.client_test_idcs}
        if self.val_frac > 0:
            entry.update({'client_val_idcs': self.client_val_idcs})
        with open(os.path.join(out_dir_path, 'client_div_info.pt'), 'wb') as f:
            pickle.dump(entry, f)


        self.data, self.targets = np.concatenate(self.data, axis=0), np.concatenate(self.targets, axis=0)

        entry = {'data': self.data, 'targets':self.targets}
       
        with open(os.path.join(out_dir_path, 'dataset.pt'), 'wb') as f:
            pickle.dump(entry, f)    

            
    def read_data(self):

        dir_path = os.path.join(self.data_dir,  "processed")  

        # if self.train == True:
        #     mod = "train"
        # else:
        #     mod = "test"
            
        with open(os.path.join(dir_path, 'dataset.pt'), 'rb') as f:
            entry = pickle.load(f)
        self.data, self.targets = entry['data'], entry['targets']   
        with open(os.path.join(dir_path, 'client_div_info.pt'), 'rb') as f:
            entry = pickle.load(f)  
        self.n_sample, self.client_train_idcs, self.client_test_idcs = entry['n_sample'], entry['client_train_idcs'], entry['client_test_idcs']
        if self.val_frac > 0:
            self.client_val_idcs = entry['client_val_idcs']
          

    def __tokenize(self, text):
        tokenized_text  = np.zeros(len(text), dtype=np.longlong)
        for ii, char in enumerate(text):
            tokenized_text[ii] = self.char2idx[char]
        return tokenized_text

    def __build_mapping(self):
        self.char2idx = dict()
        for ii, char in enumerate(self.all_characters):
            self.char2idx[char] = ii

    def __preprocess_text(self, n_sample, tokenized_text):
        data = np.zeros((n_sample, self.chunk_len), dtype=np.longlong)
        targets = np.zeros((n_sample, self.chunk_len), dtype=np.longlong)
        for idx in range(n_sample):
            data[idx] = tokenized_text[idx:idx+self.chunk_len]
            targets[idx] = tokenized_text[idx+1:idx+self.chunk_len+1]
        return data, targets

    def __len__(self):
        return self.n_sample

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    
    
# class CharacterDataset(Dataset):
#     def __init__(self, file_path, chunk_len):
#         """
#         Dataset for next character prediction, each sample represents an input sequence of characters
#          and a target sequence of characters representing to next sequence of the input
#         :param file_path: path to .txt file containing the training corpus
#         :param chunk_len: (int) the length of the input and target sequences
#         """
#         self.all_characters = string.printable
#         self.vocab_size = len(self.all_characters)
#         self.n_characters = len(self.all_characters)
#         self.chunk_len = chunk_len

#         with open(file_path, 'r') as f:
#             self.text = f.read()

#         self.tokenized_text = torch.zeros(len(self.text), dtype=torch.long)

#         self.inputs = torch.zeros(self.__len__(), self.chunk_len, dtype=torch.long)
#         self.targets = torch.zeros(self.__len__(), self.chunk_len, dtype=torch.long)

#         self.__build_mapping()
#         self.__tokenize()
#         self.__preprocess_data()

#     def __tokenize(self):
#         for ii, char in enumerate(self.text):
#             self.tokenized_text[ii] = self.char2idx[char]

#     def __build_mapping(self):
#         self.char2idx = dict()
#         for ii, char in enumerate(self.all_characters):
#             self.char2idx[char] = ii

#     def __preprocess_data(self):
#         for idx in range(self.__len__()):
#             self.inputs[idx] = self.tokenized_text[idx:idx+self.chunk_len]
#             self.targets[idx] = self.tokenized_text[idx+1:idx+self.chunk_len+1]

#     def __len__(self):
#         return max(0, len(self.text) - self.chunk_len)

#     def __getitem__(self, idx):
#         return self.inputs[idx], self.targets[idx], idx
    
    
    
    
    
    
    
    
    