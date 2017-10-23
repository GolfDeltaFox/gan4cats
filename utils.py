import config
from config import conn, s3_bucket_name
import tarfile
from os import listdir, path

def save_on_s3(model_name, save_path):
    if s3_bucket_name:
        print('Creating save archive.')
        out = tarfile.open(model_name+'.tar.gz', mode='w')
        for afile in listdir(save_path):
            out.add(path.join(save_path, afile))
        out.close()
        print('Uploading...')
        with open(model_name+'.tar.gz','rb') as f:
            conn.upload(model_name+'.tar.gz', f, s3_bucket_name)
