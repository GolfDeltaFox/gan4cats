import os
import tinys3

s3_bucket_name = 'gdf-artificial-intelligence'
S3_ACCESS_KEY = os.environ['S3_ACCESS_KEY']
S3_SECRET_KEY = os.environ['S3_SECRET_KEY']
conn = tinys3.Connection(S3_ACCESS_KEY, S3_SECRET_KEY)
