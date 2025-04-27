import glob
import json
import os
import subprocess
import tarfile

def download_model_artifacts(bucket: str, output_dir: str, logger) -> int:
    """Download the SageMaker model artifacts from the specified S3 bucket.

    Args:
        bucket (str): The name of the S3 bucket.
        output_dir (str): The local directory to download the model artifacts to.
        logger: The logger object to log information and errors.

    Returns:
        int: Returns 0 on success, 1 on failure.
    """
    # Find the correct model tarball (not profiler or logs)
    try:
        logger.info(f"Searching for SageMaker model artifacts in s3://{bucket}/model/")
        result = subprocess.run(
            f"aws s3 ls s3://{bucket}/model/ --recursive | grep '/output/model.tar.gz' | sort | tail -n 1 | awk '{{print $4}}'",
            shell=True, capture_output=True, text=True, check=False
        )
        model_file = result.stdout.strip()
        
        if not model_file:
            logger.error("Could not find SageMaker model artifact (output/model.tar.gz) in S3.")
            return 1
        
        # Download the model file
        local_model_path = os.path.join(output_dir, "model.tar.gz")
        logger.info(f"Downloading model from s3://{bucket}/{model_file} to {local_model_path}")
        result = subprocess.run(
            f"aws s3 cp s3://{bucket}/{model_file} {local_model_path}",
            shell=True, check=False
        )
        if result.returncode != 0:
            logger.error("Failed to download model file.")
            return 1
        # Extract the model if it's a tar file
        if local_model_path.endswith('.tar.gz'):
            logger.info(f"Extracting {local_model_path}")
            with tarfile.open(local_model_path, 'r:gz') as tar:
                tar.extractall(path=output_dir)
            pth_files = glob.glob(os.path.join(output_dir, '*.pth'))
            for file in pth_files:
                logger.info(f"Found model file: {file}")
        logger.info("Model downloaded and extracted successfully.")
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1