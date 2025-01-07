import torch
import copy
import numpy as np

def CorruptData(Corrupt, local_models, typeAttack, var, mean, Target ,num_clients, scale):
    if typeAttack == 0:
        return local_models
    
    #CopiedModel = copy.deepcopy(local_models[np.random.randint(1, num_clients)])
    CopiedModel = copy.deepcopy(local_models[Target])

    for i in range(num_clients):
        if Corrupt[i] == 1:
            if typeAttack == 1:
                for param in local_models[i].parameters():
                    param.data = torch.rand_like(param.data) * np.sqrt(var) + mean
                #local_models[i] = torch.rand_like(local_models[i]) * std + mean
            elif typeAttack == 2:
                #local_models[i] = CopiedModel
                local_models[i].load_state_dict(copy.deepcopy(CopiedModel.state_dict()))
            elif typeAttack == 3:
                for param in local_models[i].parameters():
                    param.data.add_(torch.randn_like(param.data) * np.sqrt(var) + mean)
            elif typeAttack == 4:
                print("Hello World")
                for key, param in local_models[i].named_parameters():
                    all_benign_params = [
                        local_models[j].state_dict()[key]
                        for j in range(num_clients)
                        if Corrupt[j] == 0 
                    ]
                    
                    benign_mean = torch.mean(torch.stack(all_benign_params), dim=0)
                    benign_std = torch.std(torch.stack(all_benign_params), dim=0)

                    param.data = benign_mean + 0.08 * benign_std  # z = 0.08  For 0.5326 = CDF
            elif typeAttack == 5:
                for key, param in local_models[i].named_parameters():
                    all_benign_params = [
                        local_models[j].state_dict()[key]
                        for j in range(num_clients)
                        if Corrupt[j] == 0 
                    ]
                    
                    benign_mean = torch.mean(torch.stack(all_benign_params), dim=0)
                    param.data = -scale * benign_mean
                    param.data.add_(torch.randn_like(param.data) * np.sqrt(var) + mean)
                
    return local_models

def CorruptGeneration(percentageCorrupt, corrupt, num_clients):
    if corrupt:
        CorruptNodes = np.random.choice(num_clients, int(num_clients*percentageCorrupt), replace=False)
        Corrupt = np.zeros(num_clients)
        for i in range(num_clients):
            if i in CorruptNodes:
                Corrupt[i] = 1
    else:
        Corrupt = np.zeros(num_clients)

    print(f'IteNumber of Corrupt nodesration {len(CorruptNodes)}, Corrupt nodes: {CorruptNodes}')

    return Corrupt