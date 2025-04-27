import argparse
import boto3
import sagemaker
from sagemaker.estimator import Estimator
import time
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--role-arn', type=str, required=True, help='SageMaker execution role ARN')
    parser.add_argument('--bucket', type=str, required=False, help='S3 bucket name')
    parser.add_argument('--input-key', type=str, required=True, help='S3 key for training data')
    parser.add_argument('--output-key', type=str, required=True, help='S3 key for model output')
    parser.add_argument('--instance-type', type=str, default='ml.m5.2xlarge', help='SageMaker instance type')
    parser.add_argument('--image-uri', type=str, default='', help='(Optional) Custom Docker image URI')
    parser.add_argument('--job-name', type=str, default=f'stockai-train-{int(time.time())}', help='SageMaker training job name')
    args = parser.parse_args()

    # Get bucket from environment variable if not provided as argument
    bucket = args.bucket
    if not bucket:
        bucket = os.environ.get('S3_BUCKET')
        if not bucket:
            raise ValueError("S3 bucket must be provided either via --bucket argument or S3_BUCKET environment variable")
    
    print(f"Using S3 bucket: {bucket}")

    session = sagemaker.Session()
    s3_input = f's3://{bucket}/{args.input_key}'
    s3_output = f's3://{bucket}/{args.output_key}'

    if args.image_uri:
        estimator = Estimator(
            image_uri=args.image_uri,
            role=args.role_arn,
            instance_count=1,
            instance_type=args.instance_type,
            output_path=s3_output,
            sagemaker_session=session,
            hyperparameters={
                'epochs': 5,
                'batch-size': 32,
                'lr': 1e-4
            }
        )
    else:
        # Use a built-in PyTorch estimator
        from sagemaker.pytorch import PyTorch
        estimator = PyTorch(
            entry_point='train_model.py',
            source_dir='.',
            role=args.role_arn,
            instance_count=1,
            instance_type=args.instance_type,
            framework_version='1.11.0',
            py_version='py310',
            output_path=s3_output,
            sagemaker_session=session,
            hyperparameters={
                'input-data': '/opt/ml/input/data/train/optimized_data.csv',
                'epochs': 5,
                'batch-size': 32,
                'lr': 1e-4,
                'disable_mixed_precision': 'true',
                'disable_self_attention': 'true',
                'fusion_type': 'concat'
            }
        )

    inputs = {
        'train': s3_input
    }

    # Check if the S3 input path exists and is accessible
    try:
        s3 = boto3.resource('s3')
        key = args.input_key
        s3.Object(bucket, key).load()
        print(f"✓ Confirmed input data exists at s3://{bucket}/{key}")
    except Exception as e:
        print(f"Error accessing S3 input data: {e}")
        return  # Exit if the S3 path is not accessible

    print(f"Launching SageMaker training job: {args.job_name}")
    estimator.fit(inputs=inputs, job_name=args.job_name, wait=True)
    print(f"Training job {args.job_name} complete. Model saved to {s3_output}")
    
    # Upload job info to S3 for easy model retrieval
    try:
        job_info = f"Job Name: {args.job_name}\nOutput Path: {s3_output}\nTimestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        s3_client = boto3.client('s3')
        s3_client.put_object(
            Bucket=bucket,
            Key=f"model/{args.job_name}/job_info.txt",
            Body=job_info
        )
        print(f"Job info saved to s3://{bucket}/model/{args.job_name}/job_info.txt")
    except Exception as e:
        print(f"Warning: Could not save job info to S3: {e}")

if __name__ == "__main__":
    main()
