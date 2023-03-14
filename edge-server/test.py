import requests



ret = requests.post("http://localhost:8000/session",json={'project_path':'hello','dataset':['china']})

print(ret.content)