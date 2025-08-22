"""Neural network model for AlphaZero Mills game."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .games import game


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)


class Model(nn.Module):
    """
    AlphaZero-style neural network for board games.
    
    Expected input shape: (input_channels, board_height, board_width)
    For example: (5, 7, 7) for Mills with 5 channels and 7x7 board
    
    Args:
        input_channels: Number of input channels (e.g. player pieces, opponent pieces, mask, phase info)
        board_height: Height of the board representation
        board_width: Width of the board representation  
        num_possible_moves: Total number of possible moves in the game
        num_filters: Number of convolutional filters in hidden layers (default: 128)
        policy_head_channels: Number of channels in policy head conv layer (default: 32)
        num_residual_blocks: Number of residual blocks (default: 8)
    """
    
    def __init__(
        self, 
        input_channels: int,
        board_height: int, 
        board_width: int,
        num_possible_moves: int,
        num_filters: int = 128,
        policy_head_channels: int = 32, 
        num_residual_blocks: int = 8
    ):
        super().__init__()
        
        # Store dimensions for documentation and validation
        self.input_channels = input_channels
        self.board_height = board_height
        self.board_width = board_width
        self.board_size = board_height * board_width
        self.num_possible_moves = num_possible_moves
        
        # Initial convolutional layer
        self.conv_input = nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_filters)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_residual_blocks)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(num_filters, policy_head_channels, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(policy_head_channels)
        self.policy_fc = nn.Linear(policy_head_channels * self.board_size, num_possible_moves)
        
        # Value head
        self.value_conv = nn.Conv2d(num_filters, policy_head_channels, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(policy_head_channels)
        self.value_fc1 = nn.Linear(policy_head_channels * self.board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass expecting batched input.
        
        Args:
            x: Input tensor of shape (batch_size, input_channels, board_height, board_width)
               
        Returns:
            policy: Policy probabilities of shape (batch_size, num_possible_moves)
            value: Value estimates of shape (batch_size, 1)
        """
        # Validate input shape
        expected_shape = (self.input_channels, self.board_height, self.board_width)
        if x.shape[1:] != expected_shape:
            raise ValueError(f"Expected input shape (batch_size, {expected_shape[0]}, {expected_shape[1]}, {expected_shape[2]}), got {x.shape}")
        
        # Initial convolution
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)  # Flatten: (batch, channels * height * width)
        policy = self.policy_fc(policy)
        policy = F.softmax(policy, dim=1)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)  # Flatten: (batch, channels * height * width)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value
    
    def get_expected_input_shape(self) -> tuple[int, int, int]:
        """Returns the expected input tensor shape: (channels, height, width)"""
        return (self.input_channels, self.board_height, self.board_width)
