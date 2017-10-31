from config import conn, s3_bucket_name, model_name, dataset_name
import tarfile
import os
import io

def initialize(dataset_name, model_name, s3_bucket_name):
    if not os.path.exists('./datasets'):
        os.makedirs('./datasets')
        print('Downloading dataset: '+str(dataset_name))
        dataset_tar = conn.get('datasets/'+dataset_name+'.tar.gz', s3_bucket_name)
        print('Extracting dataset: '+str(dataset_name))
        dataset_tar_file_like_object = io.BytesIO(dataset_tar.content)
        dataset = tarfile.open(fileobj=dataset_tar_file_like_object)
        dataset.extractall(path='./datasets/'+dataset_name)

def initialize_env():
    initialize(dataset_name, model_name, s3_bucket_name)

def main():
    initialize(dataset_name, model_name, s3_bucket_name)
if __name__ == '__main__':
    main()
