import requests
import secure_edge



instance = secure_edge.VM(ip_addr="143.248.158.197",port=8000,token="ddd")



# print(instance.health())

print(instance.fs('PCD_CT'))



project = secure_edge.Project("project")



pid = instance.upload(project)

print(pid)


ret = instance.run("11.6.2_8_ubuntu20.04_3.10",pid,['PCD_CT/train'],"Test Training")


instance.get_result()

# print(ret)