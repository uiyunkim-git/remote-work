from tulon_network_module_A import pytorch_network as NETWORK_A  # template - read only
from infer_config import *

# User Define Code Starts Here -----
from scipy import io as sio
from PIL import Image
import torch
import torchvision.transforms as transforms
import os
import torchvision


network = NETWORK_A()
network.load(
    os.path.join(
        "training",
        "5d4a86f8-067f-47b2-b22a-2f09feaad304",
        "XRALEFKDO_EPOCH_1",
    )
)


def rescale(ct):
    ct[ct < -1024.0] = -1024.0
    ct /= 4000
    return ct


def postprocessing(input_tensor, out_tensor, output_path):

    input_tensor = input_tensor.detach()
    out_tensor = out_tensor.detach().squeeze()

    out_tensor = out_tensor * 4000
    out_tensor = torch.clamp(out_tensor, -500, 500)

    input_tensor = input_tensor * 4000
    input_tensor = torch.clamp(input_tensor, -500, 500)

    torchvision.utils.save_image(
        [input_tensor.squeeze().unsqueeze(0), out_tensor.unsqueeze(0)],
        output_path,
        normalize=True,
    )


def randomize(name):
    ext = "." + name.split(".")[-1]
    return str(uuid.uuid4()) + ext


def handler(event, context):
    input_ = sio.loadmat(event["input_path"])["imdb"]["low"][0][0]
    input_ = Image.fromarray(rescale(input_))

    crop = transforms.RandomCrop(size=128)
    tf = transforms.Compose([crop, transforms.ToTensor()])
    input_ = tf(input_)

    output_ = network(input_)

    output_path = randomize("output.png")
    postprocessing(input_, output_, output_path)

    return {"success": True, "output_path": output_path}


# if __name__ == "__main__":
#     handler({"input_path": "input/L067_001.mat"}, None)

