import torch
import pickle
from functools import partial
import argparse
import os
# import models.vgg_64 as vgg

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--data_root', default='data', help='root directory for data')
parser.add_argument('--model_path', default='pretrained_models/svglp_bair.pth', help='path to model')
parser.add_argument('--log_dir', default='./results', help='directory to save generations to')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=28, help='number of frames to predict')
parser.add_argument('--num_threads', type=int, default=0, help='number of data loading threads')
parser.add_argument('--nsample', type=int, default=100, help='number of samples')
parser.add_argument('--N', type=int, default=256, help='number of samples')

opt = parser.parse_args()
os.makedirs('%s' % opt.log_dir, exist_ok=True)

# ---------------- load the models  ----------------
pickle.load = partial(pickle.load, encoding="latin1")
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")

tmp = torch.load(opt.model_path, map_location=lambda storage, loc: storage, pickle_module=pickle)

encoder = tmp['encoder']
decoder = tmp['decoder']

print(encoder.state_dict())
print(decoder.state_dict())

enc_values = encoder.state_dict()
dec_values = decoder.state_dict()

for key, value in enc_values.items():
    print(key)

# encoder = torch.load("pretrained_models/svglp_bair_enc_model.pth")
# decoder = torch.load("pretrained_models/svglp_bair_dec_model.pth")

'''
opt.g_dim = tmp['opt'].g_dim
opt.z_dim = tmp['opt'].z_dim
opt.num_digits = tmp['opt'].num_digits

encoder_ckpt = torch.load("pretrained_models/svglp_bair_enc.pth")
decoder_ckpt = torch.load("pretrained_models/svglp_bair_dec.pth")

encoder = vgg.encoder(opt.g_dim, 3)
decoder = vgg.decoder(opt.g_dim, 3)

encoder.load_state_dict(encoder_ckpt)
decoder.load_state_dict(decoder_ckpt)


frame_predictor = tmp['frame_predictor']
posterior = tmp['posterior']
prior = tmp['prior']
frame_predictor.eval()
prior.eval()
posterior.eval()

encoder = tmp['encoder']
decoder = tmp['decoder']


try:
    torch.save(encoder, "pretrained_models/svglp_bair_enc_model.pth")
    torch.save(decoder, "pretrained_models/svglp_bair_dec_model.pth")
except Exception as e:
    print(e)
    torch.save(encoder.module.features.state_dict(), "pretrained_models/svglp_bair_enc.pth")
    torch.save(decoder.state_dict(), "pretrained_models/svglp_bair_dec.pth")
for i, (name, module) in enumerate(encoder._modules.items()):
    module = recursion_change_bn(module)

for i, (name, module) in enumerate(decoder._modules.items()):
    module = recursion_change_bn(module)
    
encoder.train()
decoder.train()

encoder.eval()
decoder.eval()
'''

