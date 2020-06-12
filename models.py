# Define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        '''
        Initialize all layers of this CNN. The requirements were:
            1. This network takes in a square (same width and height), grayscale image as input
            2. It ends with a linear layer that represents the keypoints
            it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        '''
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        # First Max-pooling layer
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Fully connected layer
        self.fc1 = nn.Linear(128, 136)
        
        
    def forward(self, x):
        ''' 
        Executes feed forward
          
        Args:
            x: grayscale image as input to be analyzed by the network 
            
        Returns:
            x: output of the network
        '''
        
        ## Definition of the feedforward behavior of this model
        
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Fully connected layer
        # prep for linear layer by flattening the feature maps into feature vectors
        x = x.view(x.size(0), -1)
        # Linear Layer
        x = self.fc1(x)
        
        x = F.relu(x)
        x = self.drop1(x)
        
        # Final output
        return x
