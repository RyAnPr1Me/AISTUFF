import argparse
import boto3
import sagemaker
from sagemaker.estimator import Estimator
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--role-arn', type=str, required=True, help='SageMaker execution role ARN')
    parser.add_argument('--bucket', type=str, required=True, help='S3 bucket name')
    parser.add_argument('--input-key', type=str, required=True, help='S3 key for training data')
    parser.add_argument('--output-key', type=str, required=True, help='S3 key for model output')
    parser.add_argument('--instance-type', type=str, default='ml.m5.xlarge', help='SageMaker instance type')
    parser.add_argument('--image-uri', type=str, default='', help='(Optional) Custom Docker image URI')
    parser.add_argument('--job-name', type=str, default=f'stockai-train-{int(time.time())}', help='SageMaker training job name')
    args = parser.parse_args()

    session = sagemaker.Session()
    s3_input = f's3://{args.bucket}/{args.input_key}'
    s3_output = f's3://{args.bucket}/{args.output_key}'

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
        # Use a built-in PyTorch estimator as an example
        from sagemaker.pytorch import PyTorch
        estimator = PyTorch(
            entry_point='train_model.py',
            source_dir='.',
            role=args.role_arn,
            instance_count=1,
            instance_type=args.instance_type,
            framework_version='1.13.1',
            py_version='py38',
            output_path=s3_output,
            sagemaker_session=session,
            hyperparameters={
                'input-data': '/opt/ml/input/data/train/optimized_data.csv',
                'epochs': 5,
                'batch-size': 32,
                'lr': 1e-4
            }
        )

    inputs = {
        'train': s3_input
    }

    print(f"Launching SageMaker training job: {args.job_name}")
    estimator.fit(inputs=inputs, job_name=args.job_name, wait=True)
    print(f"Training job {args.job_name} complete. Model saved to {s3_output}")

if __name__ == "__main__":
    main()
