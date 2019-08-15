import torch

t = torch.load('./tag2pix_256.pkl')
tt = {'G': t['G']}
torch.save(tt, './tag2pix_256_stripped.pkl')
t = torch.load('./tag2pix_512.pkl')
tt = {'G': t['G']}
torch.save(tt, './tag2pix_512.stripped.pkl')