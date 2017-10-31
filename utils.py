import config
from config import conn, s3_bucket_name
import tarfile
from os import listdir, path
import threading
import os

def upload_s3(conn, file_name, s3_bucket_name):
    with open(file_name,'rb') as f:
        conn.upload(file_name, f, s3_bucket_name)

def save_on_s3(model_name, save_path):
    if s3_bucket_name:
        print('Creating save archive.')
        out = tarfile.open(model_name+'.tar.gz', mode='w')
        for afile in listdir(save_path):
            out.add(path.join(save_path, afile))
        out.close()
        print('Uploading...')
        thread = threading.Thread(target=upload_s3, args=(conn, model_name+'.tar.gz', s3_bucket_name))
        thread.start()
