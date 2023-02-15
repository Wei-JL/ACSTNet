import warnings
import torch
import torchvision
import paddle

warnings.filterwarnings("ignore", category=DeprecationWarning)

print("torch version:", torch.__version__)
print("cuda version:", torch.version.cuda)
print("cudnn version:", torch.backends.cudnn.version())

print("torchversion -V:", torchvision.__version__)
print("==============================================")
print("paddle.__version__ :: ", paddle.__version__)
print("paddleversion :: ", paddle.version.cuda())
paddle.utils.run_check()
