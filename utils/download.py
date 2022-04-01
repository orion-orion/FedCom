'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2022-02-21 14:55:08
LastEditors: ZhangHongYu
LastEditTime: 2022-02-21 18:09:10
'''
import urllib
import urllib.request
import urllib.error
from torch.utils.model_zoo import tqdm
try:
    from torchvision.version import __version__ as __vision_version__   # noqa: F401
except ImportError:
    __vision_version__ = "undefined"
import numpy as np
from torch.utils.data import Dataset
import os
import torch
from typing import Any, Callable, List, Optional
import hashlib

USER_AGENT = os.environ.get(
    "TORCHVISION_USER_AGENT",
    f"pytorch-{torch.__version__}/vision-{__vision_version__}"
)

def urlretrieve(url: str, filename: str, chunk_size: int = 1024) -> None:
    with open(filename, "wb") as fh:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT})) as response:
            with tqdm(total=response.length) as pbar:
                for chunk in iter(lambda: response.read(chunk_size), ""):
                    if not chunk:
                        break
                    pbar.update(chunk_size)
                    fh.write(chunk)

def check_integrity(fpath: str, md5: Optional[str] = None) -> bool:
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)

def check_md5(fpath: str, md5: str, **kwargs: Any) -> bool:
        return md5 == calculate_md5(fpath, **kwargs)

def calculate_md5(fpath: str, chunk_size: int = 1024 * 1024) -> str:
        md5 = hashlib.md5()
        with open(fpath, 'rb') as f:
            for chunk in iter(lambda: f.read(chunk_size), b''):
                md5.update(chunk)
        return md5.hexdigest()
    

def download_url(
    url: str, root: str, filename: Optional[str] = None, md5: Optional[str] = None) -> None:
    """Download a file from a url and place it in root.

    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
        max_redirect_hops (int, optional): Maximum number of redirect hops allowed
    """
    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    # check if file is already present locally
    if check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
        return

    # download the file
    try:
        print('Downloading ' + url + ' to ' + fpath)
        urlretrieve(url, fpath)
    except (urllib.error.URLError, IOError) as e:  # type: ignore[attr-defined]
        if url[:5] == 'https':
            url = url.replace('https:', 'http:')
            print('Failed download. Trying https -> http instead.'
                ' Downloading ' + url + ' to ' + fpath)
            urlretrieve(url, fpath)
        else:
            raise e
    # check integrity of downloaded file
    if not check_integrity(fpath, md5):
        raise RuntimeError("File not found or corrupted.")