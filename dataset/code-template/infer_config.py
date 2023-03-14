import os
import s3fs
import uuid
import builtins

# os.environ.update(
#     {
#         "USER_S3_BUCKET_PREFIX": "staging-drive-codecommit-alt/e11b6855-47c6-4ca7-a815-a21bb8418698/김의연바보",
#         "INPUT_S3_BUCKET_PREFIX": "staging-drive-codecommit-alt/e11b6855-47c6-4ca7-a815-a21bb8418698",
#         "OUTPUT_S3_BUCKET_PREFIX": "staging-drive-codecommit-alt/e11b6855-47c6-4ca7-a815-a21bb8418698/output",
#     }
# )


# os.environ.update(
#     {
#         "USER_S3_BUCKET_PREFIX": "staging-drive-codecommit-alt/e11b6855-47c6-4ca7-a815-a21bb8418698/김의연바보",
#         "INPUT_S3_BUCKET_PREFIX": "staging-drive-codecommit-alt/e11b6855-47c6-4ca7-a815-a21bb8418698",
#         "OUTPUT_S3_BUCKET_PREFIX": "staging-drive-codecommit-alt/e11b6855-47c6-4ca7-a815-a21bb8418698/output",
#     }
# )

open_bak = builtins.open


def s3fs_open_overrides(file, mode=None, **kwargs):

    if file.startswith("training"):
        bucket_prefix = os.environ.get("USER_S3_BUCKET_PREFIX")
        fs = s3fs.S3FileSystem()
        fs.invalidate_cache()
        temp_filename = f"/tmp/{str(uuid.uuid4())}"
        os.makedirs(os.path.dirname(temp_filename), exist_ok=True)
        with fs.open(f"{bucket_prefix}/{file}", "rb") as s3f:
            with open_bak(temp_filename, "wb") as f:
                f.write(s3f.read())
        file = open_bak(temp_filename, "rb")
        os.remove(temp_filename)
        print(f"s3: {file}")
        return file
    elif file.startswith("input"):
        bucket_prefix = os.environ.get("INPUT_S3_BUCKET_PREFIX")
        fs = s3fs.S3FileSystem()
        fs.invalidate_cache()
        temp_filename = f"/tmp/{str(uuid.uuid4())}"
        os.makedirs(os.path.dirname(temp_filename), exist_ok=True)
        with fs.open(f"{bucket_prefix}/{file}", "rb") as s3f:
            with open_bak(temp_filename, "wb") as f:
                f.write(s3f.read())
        file = open_bak(temp_filename, "rb")
        os.remove(temp_filename)
        print(f"s3: {file}")
        return file
    elif mode is not None:
        if mode == "wb" or mode == "w" or mode == "w+b":
            bucket_prefix = os.environ.get("OUTPUT_S3_BUCKET_PREFIX")
            fs = s3fs.S3FileSystem()
            fs.invalidate_cache()
            return fs.open(f"{bucket_prefix}/{file}", "wb")

    if mode is not None:
        return open_bak(file, mode, **kwargs)
    else:
        return open_bak(file, **kwargs)


builtins.open = s3fs_open_overrides

