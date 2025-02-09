from fastapi import FastAPI 
from fastapi.staticfiles import StaticFiles
import os
from dotenv import load_dotenv

load_dotenv()

BUCKET_NAME = os.getenv("BUCKET_NAME","STORAGE") 


ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
print("ROOT_PATH ->",ROOT_PATH)

storage_mount_app = FastAPI()
storage_mount_app.mount(f"{ROOT_PATH}/{BUCKET_NAME}", StaticFiles(directory=BUCKET_NAME, check_dir=False), name=BUCKET_NAME)
