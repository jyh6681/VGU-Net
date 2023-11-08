import numpy as np
import torch
import copy
adj = [[1,2,3,4],
       [5,6,7,8],
       [9,10,11,12]]
# re = torch.tensor([1, 2, 3, 4, 5,  6])
# result = re.view(3,2)
re = torch.tensor([[1, 2],
        		   [3, 4],
                   [5, 6]])
re = torch.unsqueeze(re,0)
re = torch.unsqueeze(re,0)

bb=copy.deepcopy(re)
bb = bb.view(1,1,-1)

re = re.view(1,1,-1)
print("re:",re,re.shape)
re = re.permute(0,2,1)
result = torch.bmm(re,bb)
print("after:",re,re.shape,bb,result)
re = re.reshape(1,1,3,2)
result = result.reshape(1,1,6,6)
print("reshape:",re,re.shape)


