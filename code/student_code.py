#from msilib.schema import Patch
from tracemalloc import start
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.modules.module import Module
from torch.nn.functional import fold, unfold
from torchvision.utils import make_grid
import math

from einops import rearrange
from einops.layers.torch import Rearrange

from utils import resize_image
import custom_transforms as transforms
from custom_blocks import PatchEmbed, TransformerBlock, trunc_normal_


#################################################################################
# You will need to fill in the missing code in this file
#################################################################################


#################################################################################
# Part I: Understanding Convolutions
#################################################################################
class CustomConv2DFunction(Function):
    @staticmethod
    def forward(ctx, input_feats, weight, bias, stride=1, padding=0):
        """
        Forward propagation of convolution operation.
        We only consider square filters with equal stride/padding in width and height!

        Args:
          input_feats: input feature map of size N * C_i * H * W
          weight: filter weight of size C_o * C_i * K * K
          bias: (optional) filter bias of size C_o
          stride: (int, optional) stride for the convolution. Default: 1
          padding: (int, optional) Zero-padding added to both sides of the input. Default: 0

        Outputs:
          output: responses of the convolution  w*x+b

        """

        # sanity check
        assert weight.size(2) == weight.size(3)
        assert input_feats.size(1) == weight.size(1)
        assert isinstance(stride, int) and (stride > 0)
        assert isinstance(padding, int) and (padding >= 0)

        # save the conv params
        kernel_size = weight.size(2)
        ctx.stride = stride
        ctx.padding = padding
        ctx.input_height = input_feats.size(2)
        ctx.input_width = input_feats.size(3)

        # make sure this is a valid convolution
        assert kernel_size <= (input_feats.size(2) + 2 * padding)
        assert kernel_size <= (input_feats.size(3) + 2 * padding)
        
        #################################################################################
        ### Start my code ###

        # Set device to cuda if available
        if not torch.cuda.is_available():
            raise Exception("Cuda is not available")
        cuda0 = torch.device('cuda:0')

        # Save dimensions for later calculations
        num_channels_out = weight.size(0)
        num_maps, num_channels_in, input_height, input_width = input_feats.shape

    
        # Pad the input
        padded_input_shape = (num_maps, num_channels_in, input_height + 2*padding, input_width + 2*padding)
        padded_input = torch.zeros(padded_input_shape, dtype=input_feats.dtype, device=cuda0)
        padded_input[:, :, padding:input_height+padding, padding:input_width+padding] = input_feats

        # Create empty output tensor with correct shape
        output_height = (input_height + 2 * padding - kernel_size) // stride + 1
        output_width = (input_width + 2 * padding - kernel_size) // stride + 1
        output = torch.zeros(num_maps, num_channels_out, output_height, output_width, dtype=input_feats.dtype, device=cuda0)

        # Perform convolution
        for c_out_idx in range(num_channels_out):
            for row_idx in range(output_height):
                start_row = row_idx * stride
                for col_idx in range(output_width):
                    start_col = col_idx * stride

                    # Need to loop over num_channels_in and kernel_size x kernel_size to find sum for each input map
                    sum_list = torch.zeros(num_maps, dtype=input_feats.dtype)
                    for n_idx in range(num_maps):
                        sum = 0.0
                        for c_in_idx in range(num_channels_in):
                            for k_row_idx in range(kernel_size):
                                for k_col_idx in range(kernel_size):
                                    cur_input = padded_input[n_idx,c_in_idx,start_row+k_row_idx,start_col+k_col_idx]
                                    cur_weight = weight[c_out_idx,c_in_idx,k_row_idx,k_col_idx]
                                    sum += cur_input * cur_weight
                        sum_list[n_idx] = sum if bias is None else sum + bias[c_out_idx]
                    output[:, c_out_idx, row_idx, col_idx] = sum_list
                    

        # Save variables for BPROP
        ctx.save_for_backward(padded_input, weight, bias)

        ### End my code ###
        #################################################################################

        return output
        
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward propagation of convolution operation

        Args:
          grad_output: gradients of the outputs

        Outputs:
          grad_input: gradients of the input features
          grad_weight: gradients of the convolution weight
          grad_bias: gradients of the bias term

        """

        #################################################################################
        ### Start my code ###

        # Set device to cuda if available
        if not torch.cuda.is_available():
            raise Exception("Cuda is not available")
        cuda0 = torch.device('cuda:0')

        # Helpful dimension variables
        padding = ctx.padding
        stride = ctx.stride
        padded_input, weight, bias = ctx.saved_tensors
        input_height = padded_input.size(2)
        input_width = padded_input.size(3)
        num_channels_out, num_channels_in, kernel_size, kernel_size = weight.shape
        num_maps, num_channels_out, output_height, output_width = grad_output.shape

        padded_grad = torch.zeros(padded_input.shape, dtype=padded_input.dtype, device=cuda0)
        grad_weight = torch.zeros(weight.shape, dtype=weight.dtype, device=cuda0)
        for c_out_idx in range(num_channels_out):
            for row_idx in range(output_height):
                start_row = row_idx * stride
                for col_idx in range(output_width):
                    start_col = col_idx * stride
                    
                    padded_grad[:,:,start_row:start_row+kernel_size,start_col:start_col+kernel_size] += \
                        grad_output[:, c_out_idx, row_idx, col_idx].reshape(-1,1,1,1) * \
                        weight[c_out_idx, :, 0:kernel_size, 0:kernel_size]

                    grad_weight[c_out_idx, :, 0:kernel_size, 0:kernel_size] += \
                        (padded_input[:,:,start_row:start_row+kernel_size,start_col:start_col+kernel_size] * \
                        grad_output[:, c_out_idx, row_idx, col_idx].reshape(-1,1,1,1)).sum(0)


        grad_input = padded_grad[:,:,padding:input_height-padding,padding:input_width-padding]

        ### End my code ###
        #################################################################################
        # compute the gradients w.r.t. input and params

        if bias is not None and ctx.needs_input_grad[2]:
            # compute the gradients w.r.t. bias (if any)
            grad_bias = grad_output.sum((0, 2, 3))

        return grad_input, grad_weight, grad_bias, None, None
    


custom_conv2d = CustomConv2DFunction.apply


class CustomConv2d(Module):
    """
    The same interface as torch.nn.Conv2D
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(CustomConv2d, self).__init__()
        assert isinstance(kernel_size, int), "We only support squared filters"
        assert isinstance(stride, int), "We only support equal stride"
        assert isinstance(padding, int), "We only support equal padding"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # not used (for compatibility)
        self.dilation = dilation
        self.groups = groups

        # register weight and bias as parameters
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        # initialization using Kaiming uniform
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # call our custom conv2d op
        return custom_conv2d(input, self.weight, self.bias, self.stride, self.padding)

    def extra_repr(self):
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}, padding={padding}"
        )
        if self.bias is None:
            s += ", bias=False"
        return s.format(**self.__dict__)


#################################################################################
# Part II: Design and train a network
#################################################################################
class SimpleNet(nn.Module):
    # a simple CNN for image classifcation
    def __init__(self, conv_op=nn.Conv2d, num_classes=100):
        super(SimpleNet, self).__init__()
        # you can start from here and create a better model
        self.features = nn.Sequential(
            # conv1 block: conv 7x7
            conv_op(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), # ADDED batch norm
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # ADDED additional block:
            conv_op(64, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64), # ADDED batch norm
            nn.ReLU(inplace=True),
            conv_op(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), # ADDED batch norm
            nn.ReLU(inplace=True),
            conv_op(64, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64), # ADDED batch norm
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # conv2 block: simple bottleneck
            conv_op(64, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64), # ADDED batch norm
            nn.ReLU(inplace=True),
            conv_op(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), # ADDED batch norm
            nn.ReLU(inplace=True),
            conv_op(64, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256), # ADDED batch norm
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # conv3 block: simple bottleneck
            conv_op(256, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128), # ADDED batch norm
            nn.ReLU(inplace=True),
            conv_op(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), # ADDED batch norm
            nn.ReLU(inplace=True),
            conv_op(128, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512), # ADDED batch norm
            nn.ReLU(inplace=True),
        )
        # global avg pooling + FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def reset_parameters(self):
        # init all params
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.consintat_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # you can implement adversarial training here
        # if self.training:
        #   # generate adversarial sample based on x
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SimpleViT(nn.Module):
    """
    This module implements Vision Transformer (ViT) backbone in
    "Exploring Plain Vision Transformer Backbones for Object Detection",
    https://arxiv.org/abs/2203.16527
    """

    def __init__(
        self,
        img_size=128,
        num_classes=100,
        patch_size=16,
        in_chans=3,
        embed_dim=192,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_abs_pos=True,
        window_size=4,
        window_block_indexes=(1, 3),
    ):
        """
        Args:
            img_size (int): Input image size.
            num_classes (int): Number of object categories
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path_rate (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            window_size (int): Window size for window attention blocks.
            window_block_indexes (list): Indexes for blocks using window attention.
            E.g., [0, 2] indicates the first and the third blocks will use window attention.

        Feel free to modify the default parameters here.
        """
        super(SimpleViT, self).__init__()

        if use_abs_pos:
            # Initialize absolute positional embedding with image size
            # The embedding is learned from data
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1, img_size // patch_size, img_size // patch_size, embed_dim
                )
            )
        else:
            self.pos_embed = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        ########################################################################
        ### Start my code ###
        
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        stride = (2, 2) # From page 15, A.2 Implementation details
        padding = (1, 1)

        self.patch_embed = PatchEmbed(patch_size, stride, padding, in_chans, embed_dim)
        self.transformer = TransformerBlock(embed_dim, num_heads, mlp_ratio, qkv_bias, drop_path_rate, norm_layer, act_layer, window_size)
        self.transformer_depth = depth
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        

        ### End my code ###
        ########################################################################
        # the implementation shall start from embedding patches,
        # followed by some transformer blocks

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)
        # add any necessary weight initialization here

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        ########################################################################
        ### Start my code ###
        # x starting shape: torch.Size([128, 3, 128, 128])
        x = self.patch_embed(x)# torch.Size([128, 58, 58, 192])
        pos = self.pos_embed # torch.Size([1, 8, 8, 192])

        for _ in range(self.transformer_depth):
            x = self.transformer(x)
        
        x = x.mean(dim = 1)
        x = self.head(x)
        x = x.mean(dim = 1)

        ### End my code ###
        ########################################################################
        return x

        # TODO: more efficient convolution forward implementation (maybe put on GPU?)
        # Reset vm and run nohup python ./main.py ../data --workers=1 --batch-size=64 --epochs=120 --wd 0.05 --lr 0.01 --use-vit &


# change this to your model!
default_cnn_model = SimpleNet
default_vit_model = SimpleViT

# define data augmentation used for training, you can tweak things if you want
def get_train_transforms(normalize):
    train_transforms = []
    train_transforms.append(transforms.Scale(144))
    train_transforms.append(transforms.RandomHorizontalFlip())
    train_transforms.append(transforms.RandomColor(0.15))
    train_transforms.append(transforms.RandomRotate(15))
    train_transforms.append(transforms.RandomSizedCrop(128))
    train_transforms.append(transforms.ToTensor())
    train_transforms.append(normalize)
    train_transforms = transforms.Compose(train_transforms)
    return train_transforms


# define data augmentation used for validation, you can tweak things if you want
def get_val_transforms(normalize):
    val_transforms = []
    val_transforms.append(transforms.Scale(144))
    val_transforms.append(transforms.CenterCrop(128))
    val_transforms.append(transforms.ToTensor())
    val_transforms.append(normalize)
    val_transforms = transforms.Compose(val_transforms)
    return val_transforms


#################################################################################
# Part III: Adversarial samples and Attention
#################################################################################
class PGDAttack(object):
    def __init__(self, loss_fn, num_steps=10, step_size=0.01, epsilon=0.1):
        """
        Attack a network by Project Gradient Descent. The attacker performs
        k steps of gradient descent of step size a, while always staying
        within the range of epsilon (under l infinity norm) from the input image.

        Args:
          loss_fn: loss function used for the attack
          num_steps: (int) number of steps for PGD
          step_size: (float) step size of PGD (i.e., alpha in our lecture)
          epsilon: (float) the range of acceptable samples
                   for our normalization, 0.1 ~ 6 pixel levels
        """
        self.loss_fn = loss_fn
        self.num_steps = num_steps
        self.step_size = step_size
        self.epsilon = epsilon

    def perturb(self, model, input):
        """
        Given input image X (torch tensor), return an adversarial sample
        (torch tensor) using PGD of the least confident label.

        See https://openreview.net/pdf?id=rJzIBfZAb

        Args:
          model: (nn.module) network to attack
          input: (torch tensor) input image of size N * C * H * W

        Outputs:
          output: (torch tensor) an adversarial sample of the given network
        """
        # clone the input tensor and disable the gradients
        output = input.clone()
        output.requires_grad = True # output.grad is None without this line
        input.requires_grad = False

        #################################################################################
        ### Start my code ###
        
        # Loop over the num_steps
        for _ in range(self.num_steps):
            output.retain_grad() # retain output gradient during BPROP
            # Make prediction and get least confident prediction
            prediction = model.forward(output)
            least_conf_vals, least_conf_indices = prediction.min(dim=1)
            # Calculate loss and perform BPROP
            loss_fn = self.loss_fn
            loss = loss_fn(prediction, least_conf_indices)
            loss.backward(retain_graph=True)
            # Update output using PGD
            output_grad = output.grad.data #https://pytorch.org/docs/stable/generated/torch.Tensor.grad.html
            sign = torch.sign(output_grad)
            output.data = output.data - self.step_size * sign
            # Clip output within epsilon bounds
            output.data = torch.clamp(output.data, input - self.epsilon, input + self.epsilon)
        
        #################################################################################

        return output


default_attack = PGDAttack


class GradAttention(object):
    def __init__(self, loss_fn):
        """
        Visualize a network's decision using gradients

        Args:
          loss_fn: loss function used for the attack
        """
        self.loss_fn = loss_fn

    def explain(self, model, input):
        """
        Given input image X (torch tensor), return a saliency map
        (torch tensor) by computing the max of abs values of the gradients
        given by the predicted label

        See https://arxiv.org/pdf/1312.6034.pdf

        Args:
          model: (nn.module) network to attack
          input: (torch tensor) input image of size N * C * H * W

        Outputs:
          output: (torch tensor) a saliency map of size N * 1 * H * W
        """
        # make sure input receive grads
        input.requires_grad = True
        if input.grad is not None:
            input.grad.zero_()

        #################################################################################
        ### My Code Start ###

        # (1) computing the input gradient by minimizing the loss of the predicted label
        # (most confident prediction);
        input.retain_grad() # retain output gradient during BPROP

        # Make prediction and get most confident prediction
        prediction = model.forward(input)
        most_conf_vals, most_conf_indices = prediction.max(dim=1)
        # Calculate loss and perform BPROP
        loss_fn = self.loss_fn
        loss = loss_fn(prediction, most_conf_indices)
        loss.backward(retain_graph=True)
        # (2) taking the absolute values of the gradients;
        input_grad_abs = torch.abs(input.grad) # https://pytorch.org/docs/stable/generated/torch.abs.html
        # (3) pick the maximum values across three color channels
        output, output_indices = input_grad_abs.max(dim=1, keepdim=True)

        ### My Code End ###
        #################################################################################

        return output


default_attention = GradAttention


def vis_grad_attention(input, vis_alpha=2.0, n_rows=10, vis_output=None):
    """
    Given input image X (torch tensor) and a saliency map
    (torch tensor), compose the visualziations

    Args:
      input: (torch tensor) input image of size N * C * H * W
      output: (torch tensor) input map of size N * 1 * H * W

    Outputs:
      output: (torch tensor) visualizations of size 3 * HH * WW
    """
    # concat all images into a big picture
    input_imgs = make_grid(input.cpu(), nrow=n_rows, normalize=True)
    if vis_output is not None:
        output_maps = make_grid(vis_output.cpu(), nrow=n_rows, normalize=True)

        # somewhat awkward in PyTorch
        # add attention to R channel
        mask = torch.zeros_like(output_maps[0, :, :]) + 0.5
        mask = output_maps[0, :, :] > vis_alpha * output_maps[0, :, :].mean()
        mask = mask.float()
        input_imgs[0, :, :] = torch.max(input_imgs[0, :, :], mask)
    output = input_imgs
    return output


default_visfunction = vis_grad_attention
