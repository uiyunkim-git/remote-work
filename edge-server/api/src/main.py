"""
FastAPI routes for edge api server
"""

import os
import docker
from docker.types import Mount
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from .schema import TrainingBody
from .config import *
import uuid
import uvicorn

app = FastAPI()


@app.post("/session")
def training(body: TrainingBody):
    """
    Start training by creating training container
    """
    body_dict = body.dict()

    project_path = body_dict["project_path"]
    dataset = body_dict["dataset"]
    session_id = str(uuid.uuid4())
    # try:
    client = docker.DockerClient(base_url="unix://var/run/docker.sock")


    mounts = [
        # Mount(
        #     target="/project",
        #     type="bind",
        #     source=f"{HOST_PROJECT_DIR}/{project_path}",
        #     read_only=False,
        # ),
        # Mount(
        #     target="/log",
        #     type="bind",
        #     source=f"{HOST_TRAINING_DIR}/{training_id}",
        #     read_only=False,
        # ),
    ]

    for data in dataset:
        mounts.append(
            Mount(
                target=f"/datasets/{data}",
                type="bind",
                source=f"{HOST_DATASET_DIR}/{data}",
                read_only=False,
            )
        )

    device_request = [
        {
            "Driver": "nvidia",
            "Capabilities": [
                ["gpu"],
                ["nvidia"],
                ["compute"],
                ["compat32"],
                ["graphics"],
                ["utility"],
                ["video"],
                ["display"],
            ],
            "Count": -1,
        }
    ]

    pypi_bridge = client.networks.list(names=["pypinetwork"])[0]
    result_bridge = client.networks.list(names=["apinetwork"])[0]

    env_var = [
        f"SESSION_ID={session_id}",
    ]

    if ENABLE_GPU:
        training_container = client.containers.run(
            name=session_id,
            privileged=True,
            image=f"{TRAINING_IMAGE}",
            network_disabled=False,
            network=pypi_bridge.id,
            device_requests=device_request,
            detach=True,
            environment=env_var,
            mounts=mounts,
            cpu_count=4,
            mem_limit="16g",
            labels={"id": session_id},
        )
    else:
        training_container = client.containers.run(
            name=session_id,
            privileged=True,
            image=f"{TRAINING_IMAGE}",
            network_disabled=False,
            network=pypi_bridge.id,
            detach=True,
            environment=env_var,
            mounts=mounts,
            cpu_count=4,
            mem_limit="16g",
            labels={"id": session_id},
        )


    ret = result_bridge.connect(training_container)


    return {"success": True, "id": training_container.id, "result_path": session_id}

    # except Exception as exception:
    #     return JSONResponse(status_code=514, content="Failed to start training")



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="trace")