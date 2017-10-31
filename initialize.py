from config import conn, s3_bucket_name, model_name, dataset_name
import tarfile
import os
import io
import requests

def get_archive_and_extract(path, file_name, s3_bucket_name, extension='.tar.gz'):
    if not os.path.exists(path):
        # os.makedirs(path)
        full_path = os.path.join(path, file_name+extension)
        print('Downloading: '+s3_bucket_name+full_path)
        try:
            s3_data = conn.get(full_path, s3_bucket_name)
            print('Extracting: '+str(file_name+extension)+' to '+str(path))
            tar_file_like = io.BytesIO(s3_data.content)
            tar_obj = tarfile.open(fileobj=tar_file_like)
            tar_obj.extractall(path=path+dataset_name)
        except requests.exceptions.HTTPError:
            print('No such ressource on s3. Starting over.')
            os.makedirs(path)
    else:
        print(path+' already exists.')

def initialize(dataset_name, model_name, s3_bucket_name):
    get_archive_and_extract('./datasets', dataset_name, s3_bucket_name)
    get_archive_and_extract('./checkpoints', model_name, s3_bucket_name)

def initialize_env():
    initialize(dataset_name, model_name, s3_bucket_name)

def main():
    initialize(dataset_name, model_name, s3_bucket_name)
if __name__ == '__main__':
    main()
