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
        # The network expects input image 224x224
        
        # First Convolutional layer + Max-pooling layer
        self.conv1 = nn.Conv2d(1, 32, 5) # 1 as it's greyscale image (1 channel,  )
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Second Convolutional layer + Max-pooling layer
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool2 = nn.MaxPool2d(2, 2)

        
        # First Fully connected layer
        self.fc1 = nn.Linear(179776, 3000) 
        
        # Second Fully connected layer
        self.fc2 = nn.Linear(3000, 500) 
        
        # Third  Fully connected layer
        self.fc3 = nn.Linear(500, 136) 
        
        
    def forward(self, x):
        ''' 
        Executes feed forward
          
        Args:
            x: grayscale image as input to be analyzed by the network 
            
        Returns:
            x: output of the network
        '''
        
        ## Definition of the feedforward behavior of this model
        
        # first conv/relu + pool layer
        x = self.pool1(F.relu(self.conv1(x)))
        # second conv/relu + pool layer
        x = self.pool2(F.relu(self.conv2(x)))
        
        # Fully connected layer
        
        # prep for linear layer by flattening the feature maps into feature vectors
        x = x.view(x.size(0), -1) 
        # First fully connected layer   
        x = F.relu(self.fc1(x))
        # Second fully connected layer   
        x = F.relu(self.fc2(x))
        # Third fully connected layer   
        x = self.fc3(x)
        
        # Final output
        return x
