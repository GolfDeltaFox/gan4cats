import config
from config import conn, s3_bucket_name

def save_on_s3(model_name, save_path):
    if s3_bucket_name:
        fi = open(save_path+'.index','rb')
        fm = open(save_path+'.meta','rb')
        conn.upload(model_name+'.index',fi, s3_bucket_name)
        conn.upload(model_name+'.meta',fm, s3_bucket_name)
