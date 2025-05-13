import torch
import copy
import numpy as np

def CorruptData(Corrupt, node_model_s, typeAttack, var, mean, Target ,num_clients, scale):
    if typeAttack == 0:
        return node_model_s
    
    #CopiedModel = copy.deepcopy(local_models[np.random.randint(1, num_clients)])
    CopiedModel = copy.deepcopy(node_model_s[Target])

    for i in range(num_clients):
        if Corrupt[i] == 1:
            if typeAttack == 1: # Random Attack
                for param in node_model_s[i].parameters():
                    param.data = torch.rand_like(param.data) * np.sqrt(var) + mean
                #local_models[i] = torch.rand_like(local_models[i]) * std + mean
            elif typeAttack == 2: # Copied Attack
                #local_models[i] = CopiedModel
                node_model_s[i].load_state_dict(copy.deepcopy(CopiedModel.state_dict()))
            elif typeAttack == 3: # Added random nois attack
                for param in node_model_s[i].parameters():
                    param.data.add_(torch.randn_like(param.data) * np.sqrt(var) + mean)
            elif typeAttack == 4: 
                # print("Hello World")
                for key, param in node_model_s[i].named_parameters():
                    all_benign_params = [
                        node_model_s[j].state_dict()[key]
                        for j in range(num_clients)
                        if Corrupt[j] == 0 
                    ]
                    
                    benign_mean = torch.mean(torch.stack(all_benign_params), dim=0)
                    benign_std = torch.std(torch.stack(all_benign_params), dim=0)

                    param.data = benign_mean + 0.08 * benign_std  # z = 0.08  For 0.5326 = CDF
            elif typeAttack == 5:
                for key, param in node_model_s[i].named_parameters():
                    # all_benign_params = [
                    #     node_model_s[j].state_dict()[key]
                    #     for j in range(num_clients)
                    #     if Corrupt[j] == 0 
                    # ]
                    
                    # benign_mean = torch.mean(torch.stack(all_benign_params), dim=0)
                    param.data = -scale * param.data
                    # param.data.add_(torch.randn_like(param.data) * np.sqrt(var) + mean)
                
    return node_model_s

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