# potsdam-batch-exp

Each experiment notebook contains an `EXPERIMENT_NAME` variable and will write all data related to that experiment to the dir `s3://raster-vision-ahassan/potsdam/experiments/output/<EXPERIMENT_NAME>/`.

## Running on AWS Batch
1. Upload notebook to s3
2. Create a Batch job with
    - **Job definition**: ahassanAWSBatch:2
    - **Command**: 
      ```
      ./download_run_upload_jupyter.sh <S3 path to notebook> <S3 path to dir where executed notebook will be uploaded>
      ```
      Example
      ```
      ./download_run_upload_jupyter.sh s3://raster-vision-ahassan/potsdam/experiments/notebooks/ss_e_deeplab101.ipynb s3://raster-vision-ahassan/potsdam/experiments/output/ss_e_deeplab101
      ```
