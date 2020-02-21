
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import torch
import numpy as np
image = Image.open("images/zebra.jpg")
print("Image shape:", image.size)

model = torchvision.models.resnet18(pretrained=True)
print(model)
first_conv_layer = model.conv1
print("First conv layer weight shape:", first_conv_layer.weight.shape)
print("First conv layer:", first_conv_layer)

# Resize, and normalize the image with the mean and standard deviation
image_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image = image_transform(image)[None]
print("Image shape:", image.shape)

activation = first_conv_layer(image)
print("Activation shape:", activation.shape)


def torch_image_to_numpy(image):
    """
    Function to transform a pytorch tensor to numpy image
    Args:
        image: shape=[3, height, width]
    Returns:
        iamge: shape=[height, width, 3] in the range [0, 1]
    """
    # Normalize to [0 - 1.0]
    image = image.detach().cpu() # Transform image to CPU memory (if on GPU VRAM)
    image = image - image.min()
    image = image / image.max()
    image = image.numpy()
    if len(image.shape) == 2: # Grayscale image, can just return
        return image
    assert image.shape[0] == 3, "Expected color channel to be on first axis. Got: {}".format(image.shape)
    image = np.moveaxis(image, 0, 2)
    return image

print(list(model.children()))
print(len(list(model.children())))

model = list(model.children())
model = model[0:-2]
input_img = image
for i in range(len(model)):
    input_img = model[i](input_img)

activation = input_img

print(activation.shape)
indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
for i in range(len(indices)):
    indice = indices[i]
    img_np = torch_image_to_numpy(activation[0,indice,:,:])
    plt.subplot(2,5, i + 1) 
    imgplot = plt.imshow(img_np)

# for i in range(len(indices)):
#     indice = indices[i]
#     img_np = torch_image_to_numpy(first_conv_layer.weight[indice,:,:,:])
#     plt.subplot(2,5, i + 6) 
#     imgplot = plt.imshow(img_np)

plt.subplots_adjust(wspace = 0.6)
plt.subplots_adjust(hspace = 0.05)
plt.colorbar()
plt.savefig("task_4c.png")
plt.show()
