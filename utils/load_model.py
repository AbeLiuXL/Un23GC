import torch
def load_pre_model(model,model_path,str_h='flow_net.'):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path,map_location=torch.device('cpu'))['state_dict']
    # pretrained_dict = torch.load(model_path)
    new_pretrained_dict = {}
    for k in model_dict:
        new_pretrained_dict[k] = pretrained_dict[str_h + k]  # tradition training
    model_dict.update(new_pretrained_dict)
    model.load_state_dict(model_dict)
    return model