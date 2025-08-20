from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Visit https://0.0.0.0:8001/docs to see the API documentation and test the endpoints

@app.get("/")
async def read_root():
    return {"result": "Plugin Template backend is working!"}

@app.post("/copy-to-volume")
async def copy_to_volume_call(file_path: str):
    # Copy a file from an absolute path on the host file system
    # to the plugin folder in the volume ("/volume/PLUGIN_NAME/")
    # sourcePath: absolute path of the file on the host file system
    # targetPath: relative path of the file in the plugin folder in the volume
    files = [
        {"sourcePath": file_path, "targetPath": "./"},
    ]

    await copy_to_volume(files)

    return {"result": "File copied to volume!"}


### Volume server communication ###
PLUGIN_NAME = "plugin-template"
VOLUME_SERVER_URL = "http://host.docker.internal:3001"

async def copy_to_volume(files):
    data = {
        "volumeName": "ouroboros-volume",
        "pluginFolderName": PLUGIN_NAME,
        "files": files,
    }

    return await request_volume_server("copy-to-volume", data)


async def copy_to_host(files):
    data = {
        "volumeName": "ouroboros-volume",
        "pluginFolderName": PLUGIN_NAME,
        "files": files,
    }

    return await request_volume_server("copy-to-host", data)


async def clear_plugin_folder():
    data = {
        "volumeName": "ouroboros-volume",
        "pluginFolderName": PLUGIN_NAME,
    }

    return await request_volume_server("clear-plugin-folder", data)


async def request_volume_server(path, data):
    url = f"{VOLUME_SERVER_URL}/{path}"
    try:
        result = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=data,
        )
        if not result.ok:
            return False, result.text
        else:
            return True, ""
    except Exception as error:
        return False, str(error)