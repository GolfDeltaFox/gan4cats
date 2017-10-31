import os
import tinys3

dataset_name = os.environ['DATASET_NAME']
s3_bucket_name = os.environ['S3_BUCKET_NAME']
S3_ACCESS_KEY = os.environ['S3_ACCESS_KEY']
S3_SECRET_KEY = os.environ['S3_SECRET_KEY']
model_name = os.environ['TF_MODEL_NAME']

conn = tinys3.Connection(S3_ACCESS_KEY, S3_SECRET_KEY, tls=True)
