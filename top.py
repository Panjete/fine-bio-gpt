import sys
from train import train
from test import test

mode = sys.argv[1]

if(mode=="train"):
    print("Training started!")
    train(sys.argv[2], sys.argv[3]) # [path_to_data] [path_to_save]
else:
    print("Testing started!")
    test(sys.argv[2], sys.argv[3], sys.argv[4])  # [path_to_data]  [path_to_model] [path_to_save]


