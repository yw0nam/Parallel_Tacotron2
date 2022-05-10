import torch
import torch.nn as nn
from torch.nn import functional as F
# ====================================== #
#             Basic Layers               #
# ====================================== #

class LightweightConv(nn.Module):
    def __init__(
        self,
        num_channels,
        kernel_size,
        padding_l,
        weight_softmax,
        num_heads,
        weight_dropout,
        stride=1,
        dilation=1,
        bias=True,
                ):
        super(LightweightConv, self).__init__()
        
        self.channels = num_channels
        self.heads = num_heads
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding_l
        self.dilation = dilation
        self.dropout_p = weight_dropout
        self.bias = bias
        self.weight_softmax = weight_softmax
        
        self.weights = nn.Parameter(torch.Tensor(self.heads, 1, self.kernel_size), requires_grad=True)
        
        self.kernel_softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(self.dropout_p)
        
        if self.bias:
            self.bias_weights = nn.Parameter(torch.randn(self.heads))

        self.reset_parameters()    
            
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        if self.bias_weights is not None:
            nn.init.constant_(self.bias_weights, 0.)
            
    def forward(self, x):

        x = x.contiguous().transpose(1, 2)
        # x.shape = [batchsize, channel, width]
        batch_size, in_channel, width = x.shape
        
        if self.weight_softmax:
            weights = self.kernel_softmax(self.weights)
        else:
            weigths = self.weights
            
        weigths = self.dropout(weights)
        
        x = x.reshape(-1, self.heads, width)
        
        if self.bias:
            output = F.conv1d(x, weigths, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.heads, bias=self.bias_weights)
        else:
            output = F.conv1d(x, weigths, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.heads)
        
        output = output.reshape(batch_size, -1, width).contiguous().transpose(1, 2)
        
        return output
    
class LinearNorm(nn.Module):
    """
    Linear Module
    """

    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        """
        :param in_dim: dimension of input
        :param out_dim: dimension of output
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init))
        if bias:
            nn.init.constant_(self.linear_layer.bias, 0.0)
    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm1D(nn.Module):
    """
    Convolution Module
    """

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1,
                 padding=0, dilation=1, bias=True, w_init='linear'):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(ConvNorm1D, self).__init__()

        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation,
                              bias=bias)

        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init))
        if bias:
            nn.init.constant_(self.conv.bias, 0.0)
            
    def forward(self, x):
        x = self.conv(x)
        return x


class FFN(nn.Module):
    """
    Feed Forward Network
    """

    def __init__(self, hidden_size):
        super(FFN, self).__init__()
        self.w_1 = LinearNorm(hidden_size, hidden_size * 4)
        self.w_2 = LinearNorm(hidden_size * 4, hidden_size)

    def forward(self, input_):

        x = self.w_2(torch.relu(self.w_1(input_)))
        return x

# ====================================== #
#                Blocks                  #
# ====================================== #

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dropout, kernel_size=5, activation=nn.ReLU()):
        super(ConvBlock, self).__init__()
        
        self.layer = nn.Sequential(
            ConvNorm1D(
                in_channel,
                out_channel,
                kernel_size=kernel_size,
                stride=1,
                padding="same",
                dilation=1),
            nn.BatchNorm1d(out_channel),
            activation
        )
        
        self.dropout = nn.Dropout(dropout)
        # self.layer_norm = nn.LayerNorm(out_channel)
        
    def forward(self, x, mask=None):
        x = self.dropout(self.layer(x.contiguous().transpose(1, 2)))
        x = x.contiguous().transpose(1, 2)
        # x = self.layer_norm(x.contiguous().transpose(1, 2))
        if mask is not None:
            # x = x.masked_fill(mask.lt(1).unsqueeze(-1), 0)
            x = x.masked_fill(mask.unsqueeze(-1), 0)
        return x
        
class TransformerEncoder(nn.Module):
    """
    Encoder Block
    """

    def __init__(self, hidden_size, num_heads, dropout_p):
        """
        Multihead Attention(MHA) : Q, K and V are equal
        """
        super(TransformerEncoder, self).__init__()
        self.MHA = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.MHA_dropout = nn.Dropout(dropout_p)
        self.MHA_norm = nn.LayerNorm(hidden_size)
        self.FFN = FFN(hidden_size)
        self.FFN_norm = nn.LayerNorm(hidden_size)
        self.FFN_dropout = nn.Dropout(dropout_p)

    def forward(self, input_, mask):
        x = input_
        x, attn = self.MHA(query=x, key=x, value=x, key_padding_mask=mask)
        x = self.MHA_norm(input_ + self.MHA_dropout(x))
        x = self.FFN_norm(x + self.FFN_dropout(self.FFN(x)))
        return x, attn


class LConvBlock(nn.Module):
    """ Lightweight Convolutional Block """

    def __init__(self, hidden_size, kernel_size, num_heads, dropout, stride=1, weight_softmax=True):
        super(LConvBlock, self).__init__()
        
        padding_l = (
            kernel_size // 2
            if kernel_size % 2 == 1
            else ((kernel_size - 1) // 2, kernel_size // 2)
        )

        self.act_linear = LinearNorm(
            hidden_size, 2 * hidden_size, bias=True)
        self.act = nn.GLU()

        self.conv_layer = LightweightConv(
            hidden_size,
            kernel_size,
            padding_l=padding_l,
            weight_softmax=weight_softmax,
            num_heads=num_heads,
            weight_dropout=dropout,
            stride=stride
        )

        self.FFN = FFN(hidden_size)
        self.FFN_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, mask=None):

        # x.shape = [batch_size, time_step, channel]
        residual = x
        x = self.act_linear(x)
        x = self.act(x)
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(2), 0)
        x = self.conv_layer(x)
        x = residual + x

        residual = x
        x = self.FFN(x)
        x = residual + x
        x = self.FFN_norm(x)

        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(2), 0)
    
        return x


class SwishBlock(nn.Module):
    """ Swish Block """

    def __init__(self, in_channels, hidden_dim, out_channels):
        super(SwishBlock, self).__init__()
        self.layer = nn.Sequential(
            LinearNorm(in_channels, hidden_dim, bias=True),
            nn.SiLU(),
            LinearNorm(hidden_dim, out_channels, bias=True),
            nn.SiLU(),
        )

    def forward(self, S, E, V):

        out = torch.cat([
            S.unsqueeze(-1),
            E.unsqueeze(-1),
            V.unsqueeze(1).expand(-1, E.size(1), -1, -1),
        ], dim=-1)
        out = self.layer(out)

        return out
