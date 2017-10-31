import config
from config import conn, s3_bucket_name
import tarfile
from os import listdir, path
import threading
import os
import io

def upload_s3(conn, file_name, data, s3_bucket_name):
        bytes_like = io.BytesIO(data)
        conn.upload(file_name, bytes_like, s3_bucket_name)

def test_path_to_archive_bytes(fpath):
    print('Creating archive of: '+ fpath)
    bytes_like = io.BytesIO()
    tar = tarfile.open(mode='w', fileobj=bytes_like)
    for afile in listdir(fpath):
        tar.add(path.join(fpath, afile))
    tar.close()
    val = bytes_like.getvalue()
    bytes_like.close()
    return val

def save_on_s3(model_name, fpath):
    if s3_bucket_name:
        out = test_path_to_archive_bytes(fpath)
        full_path = path.join(fpath, model_name+'.tar.gz')
        print('Uploading to '+str(full_path)+'')
        thread = threading.Thread(target=upload_s3, args=(conn, full_path, out, s3_bucket_name))
        thread.start()
