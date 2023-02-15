import os
import mlflow

def log_tags():
    envlist = {}
    for key, value in os.environ.items():
        envlist[key.strip().lower()] = value.strip()
    # end-of-for
    mlflow.set_tags({
    'cvat-taskid':envlist.get('cvattaskid','none'),
    'model_type': envlist.get('model_type','yolov7'),
    'width':envlist.get('width','608'),
    'height':envlist.get('height','608'),
    'batchsize':envlist.get('batchsize','32'),
    'epochs':envlist.get('epochs','150')
    })
def log_parameters(hyp, opt):
    for k, v in hyp.items():
        mlflow.log_param(k, v)
    for key, value in vars(opt).items():
        mlflow.log_param(key, value)

def log_model(model, opt, epoch, fitness_score, best_model=False):
    if best_model:
        mlflow.pytorch.log_model(model, artifact_path="best.pt")
        mlflow.log_artifact("/temp/data.yaml",artifact_path="best.pt/data/yolov7")
        mlflow.log_artifact("/output/yolov7/weights/best.pt",artifact_path="best.pt/data/yolov7/weights")
    else:
        mlflow.pytorch.log_model(model, artifact_path="last.pt")
        mlflow.log_artifact("/temp/data.yaml",artifact_path="last.pt/data/yolov7")
        mlflow.log_artifact("/output/yolov7/weights/last.pt",artifact_path="last.pt/data/yolov7/weights") 
    # mlflow.pytorch.log_model(model, artifact_path=f"epoch{epoch+1}.pt")
    print("The model is logged at:\n%s" % (os.path.join(mlflow.get_artifact_uri(), "last.pt")))
    print("Saving model artifact on epoch ", epoch + 1)

def support_mlflow() -> bool:
    '''
    doc: https://docs.oneai.twcc.ai/s/3uxGFglX0
    '''
    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    mlflow_s3_endpoint_url = os.environ.get('MLFLOW_S3_ENDPOINT_URL')
    mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', '')

    enable_mlflow = bool(aws_access_key_id \
                            or aws_secret_access_key \
                            or mlflow_s3_endpoint_url \
                            or mlflow_tracking_uri)

    # check parameters
    if enable_mlflow:
        if not aws_access_key_id:
            enable_mlflow = False
            print('[MLflow][WARNING] will enable MLflow, '
                    'but missing the key: AWS_ACCESS_KEY_ID')

        if not aws_secret_access_key:
            enable_mlflow = False
            print('[MLflow][WARNING] will enable MLflow, '
                    'but missing the key: AWS_SECRET_ACCESS_KEY')

        if not mlflow_s3_endpoint_url:
            enable_mlflow = False
            print('[MLflow][WARNING] will enable MLflow, '
                    'but missing the key: MLFLOW_S3_ENDPOINT_URL')

        if not mlflow_tracking_uri:
            enable_mlflow = False
            print('[MLflow][WARNING] will enable MLflow, '
                    'but missing the key: MLFLOW_TRACKING_URI')

    return enable_mlflow


def get_mlflow_experiment_name():
    return os.environ.get('MLFLOW_EXPERIMENT_NAME', '')
