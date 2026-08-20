# import boto3
# import sagemaker

# # sm_client = boto3.client('sagemaker')
# model_package_group_name = "MyUniqueModelGroup"

# # sm_client.create_model_package_group(
# #     ModelPackageGroupName=model_package_group_name,
# #     ModelPackageGroupDescription="Group for my production models"
# # )

# from sagemaker.model import Model

# model = Model(
#     image_uri="your-container-image-uri",
#     model_data="s3://your-bucket/model.tar.gz",
#     role="your-iam-role-arn"
# )

# model.register(
#     content_types=["text/csv"],
#     response_types=["text/csv"],
#     inference_instances=["ml.m5.large"],
#     transform_instances=["ml.m5.large"],
#     model_package_group_name=model_package_group_name,
#     approval_status="PendingManualApproval" # Can be 'Approved' or 'Rejected'
# )


# import sagemaker
# from sagemaker.pytorch import PyTorchModel

# role = "arn:aws:iam::251801022342:role/service-role/AmazonSageMakerAdminIAMExecutionRole"

# model = PyTorchModel(
#     model_data="s3://traffic-data-bucket-123/model/model.tar.gz",
#     role=role,
#     entry_point="inference.py",
#     framework_version="1.13",
#     py_version="py39"
# )

# predictor = model.deploy(
#     instance_type="ml.m5.large",
#     initial_instance_count=1
# )

# print(predictor.endpoint_name)