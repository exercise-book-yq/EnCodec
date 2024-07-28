from model import EncodecModel
import torch
from torchsummary import summary

model = EncodecModel.my_encodec_model('/home/Datasets1/youqiang/cp_hifigan/Librispeech/encodec/v7/g_00300000')

total_params = sum(p.numel() for p in model.encoder.parameters()if p.requires_grad)
total_params += sum(p.numel() for p in model.decoder.parameters()if p.requires_grad)
# total_params += sum(p.numel() for p in model.quantizer.parameters()if p.requires_grad)
print("Total parameters (in M):", total_params / 1_000_000)
