import requests

import tarfile
import os.path
import io
import uuid

def make_tarfile(source_dir):
    buffer = io.BytesIO()
    with tarfile.open(mode= "w:gz",fileobj=buffer) as tar:
        tar.add("project/", arcname="")
    buffer.seek(0)
    return buffer

class Project:

    def __init__(self,source_dir) -> None:
        self._source_dir = source_dir


    def secure(self):
        tarfile = make_tarfile(self._source_dir)
        return tarfile

class VM:

    def __init__(self,ip_addr,port) -> None:

        self._ip_addr = ip_addr
        self._port = port
        pass


    def health(self) -> bool:
        try:
            ret = requests.get(f"http://{self._ip_addr}:{self._port}/gpu")
            if ret.status_code == 200:
                return True
            else:
                return False
        except:
            return False
        
    
    def fs(self,path,vis=True):
        try:
            ret = requests.get(f"http://{self._ip_addr}:{self._port}/storage/{path}")
            if ret.status_code == 200:

                contents = [x['filename'] for x in ret.json()['contents']]


                if vis:
                    for i, p in enumerate(path.split('/')):
                        print('--'*i + p )
                    for c in contents:
                        print('--'*(i+1) + c )
                return contents
            else:
                return False
        except:
            return False
        

    def upload(self,project:Project):
        
        try:
            ret = requests.post(f"http://{self._ip_addr}:{self._port}/upload",files={"project":project.secure().read()})
            print(ret.content)
            if ret.status_code == 200:

                return ret.json()['project_path']
            else:
                return False
        except:
            return False
        
    def run(self,image_tag,pid,data_paths,name):
        
        try:
            tid = str(uuid.uuid4())
            ret = requests.post(f"http://{self._ip_addr}:{self._port}/training",json={
                "image_tag": image_tag,
                "project_path":pid,
                "dataset":data_paths,
                "training_id":tid,
                "training_name":name
            })
            if ret.status_code == 200:
                return tid, ret.json()
            else:
                return False
        except:
            return False
        

        pass


if __name__ == '__main__':
    project = Project("project")

    project.secure()