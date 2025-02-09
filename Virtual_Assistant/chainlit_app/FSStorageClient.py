from chainlit.config import config, DEFAULT_HOST
import mimetypes
import os
from chainlit.data.base  import BaseStorageClient
from typing import  Dict, Any, Union
from aiohttp import ClientError, ClientSession as Session, ServerTimeoutError
import aiofiles
from chainlit.logger import logger

#https://github.com/Chainlit/chainlit/issues/1205

class FSStorageClient(BaseStorageClient):
    """
    Class to enable File System storage for ChainLit elements.
    """
    def __init__(self, storage_path: str, url_path: str):
        self.storage_path = storage_path
        self.url_path = url_path
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path, exist_ok=True)

        # Get serving URL
        host = config.run.host
        port = config.run.port

        if host == DEFAULT_HOST:
            self.url = f"http://localhost:{port}{os.environ.get('CHAINLIT_ROOT_PATH', '')}"
        else:
            self.url = f"http://{host}:{port}{os.environ.get('CHAINLIT_ROOT_PATH', '')}"

    async def upload_file(self, object_key: str, data: Union[bytes, str],
                          mime: str = 'application/octet-stream', overwrite: bool = True) -> Dict[str, Any]:

        try:
            # Clean file key and attempt to steal extension
            object_key, s_existing_extension = os.path.splitext(object_key)

            if s_existing_extension == "":
                # Guess extension if there is none
                s_file_extension = mimetypes.guess_extension(mime)
            else:
                s_file_extension = s_existing_extension
            s_object_key_final = object_key + s_file_extension
            s_object_key_url = s_object_key_final.replace("\\", "/")

            s_file_path = os.path.join(self.storage_path, s_object_key_final)

            # Ensure directory exists, Python does not create them automatically
            os.makedirs(os.path.dirname(s_file_path), exist_ok=True)

            # If we should not overwrite, fail if file exists
            if not overwrite and os.path.exists(s_file_path):
                return {}

            logger.debug(f"FSStorageClient, uploading file to: '{s_file_path}'")

            # Open the file in binary write mode
            async with aiofiles.open(s_file_path, "wb") as f:
                # Check if data is of type str, if yes, convert to bytes
                if isinstance(data, str):
                    data = data.encode('utf-8')
                await f.write(data)

            # Calculate URL for this file
            s_file_url = f"{self.url}/{self.url_path}/{s_object_key_url}"

            logger.debug(f"FSStorageClient, saving access URL as: '{s_file_url}'")

            return {"object_key": s_object_key_final, "url": s_file_url}

        except Exception as e:
            logger.warn(f"FSStorageClient, upload_file error: {e}")
            return {}