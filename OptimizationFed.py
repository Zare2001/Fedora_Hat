import torch
from torch import amp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import copy
import traceback
import itertools
import pickle
from pathlib import Path
# Custom imports; ensure these modules are in your PYTHONPATH
from UtilityGraph import *
from Defence import *
from Corruption import *
from UtilityMLP import *
import gc
import os
from contextlib import contextmanager
import warnings

# ───────────────────── results-persistence helpers ─────────────────────
from collections import defaultdict
from datetime import datetime

def append_result(entry, path="resultsNew.pkl"):
    """
    Append *one* experiment result to the pickle file.
    Each call writes an independent pickle frame.
    """
    with open(path, "ab") as fh:           # binary-append mode
        pickle.dump(entry, fh, protocol=pickle.HIGHEST_PROTOCOL)


def _nested_dict() -> defaultdict:          # top-level → picklable
    return defaultdict(dict)

_RESULTS = defaultdict(_nested_dict)      

def _to_plain(obj):
    "Recursively turn defaultdicts into plain dicts."
    if isinstance(obj, defaultdict):
        obj = {k: _to_plain(v) for k, v in obj.items()}
    elif isinstance(obj, dict):
        obj = {k: _to_plain(v) for k, v in obj.items()}
    return obj


def save_results(path: str = "resultsNew.pkl") -> None:
    """
    Persist global _RESULTS; merge under a timestamp if file exists.
    """
    ts_key = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # ➊ convert to plain dict to avoid defaultdict pickling woes
    clean_results = _to_plain(_RESULTS)

    payload = {ts_key: clean_results}

    p = Path(path)
    if p.exists():
        with p.open("rb") as fh:
            try:
                existing = pickle.load(fh)
            except Exception:
                existing = {}
        existing.update(payload)
        payload = existing

    tmp = p.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)  # highest proto → smaller/faster
    tmp.replace(p)

    n_priv = len(clean_results)
    n_atk  = sum(len(a) for a in clean_results.values())
    n_noise = sum(len(n) for a in clean_results.values() for n in a.values())
    print(f"Saved {n_priv=} {n_atk=} {n_noise=} to {p} under key '{ts_key}'.")



# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gc.collect()
torch.cuda.empty_cache()

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:128,expandable_segments:True"
)


@contextmanager
def cuda_guard(section: str = ""):
    """Wrap GPU-heavy blocks to catch OOM and clean up."""
    try:
        yield
    except torch.cuda.OutOfMemoryError as e:
        warnings.warn(
            f"[OOM in {section}] {e}.  Emptying cache and re-raising ...",
            RuntimeWarning
        )
        torch.cuda.empty_cache()
        gc.collect()
        raise

# class Current(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(Current, self).__init__()
        
#         self.layers = nn.Sequential(
#             nn.Linear(input_size, hidden_size),  # Layer 1
#             nn.ReLU(),
#             nn.Linear(hidden_size, hidden_size), # Layer 2
#             nn.ReLU(),
#             nn.Linear(hidden_size, hidden_size), # Layer 3
#             nn.ReLU(),
#             nn.Linear(hidden_size, hidden_size), # Layer 4
#             nn.ReLU(),
#             nn.Linear(hidden_size, hidden_size), # Layer 5
#             nn.ReLU(),
#             nn.Linear(hidden_size, hidden_size), # Layer 6
#             nn.ReLU(),
#             nn.Linear(hidden_size, num_classes)  # Output Layer (Layer 7)
#         )

#     def forward(self, x):
#         x = x.view(x.size(0), -1)  # Flatten the input
#         return self.layers(x)



class Current(nn.Module):
    """Two-layer MLP model."""
    def __init__(self, input_size, hidden_size, num_classes):
        super(Current, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class FlipLabelDataset(Dataset):
    """Dataset wrapper that flips labels according to a fixed map."""
    def __init__(self, dataset):
        self.dataset = dataset
        self.label_map = {0:3,1:4,2:7,3:5,4:8,5:0,6:9,7:6,8:2,9:1}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        flipped_label = self.label_map[label]
        return image, flipped_label


def split_dataset(dataset, num_clients):
    """Randomly split a dataset into `num_clients` subsets."""
    dataset_size = len(dataset)
    indices = np.random.permutation(dataset_size)
    data_per_client = dataset_size // num_clients
    split_sizes = [data_per_client] * num_clients
    for i in range(dataset_size % num_clients):
        split_sizes[i] += 1
    subsets = []
    start = 0
    for size in split_sizes:
        subsets.append(Subset(dataset, indices[start:start + size]))
        start += size
    return subsets

def aggregate_models(client_datasets, node_models, G, tolerance, c, max_iters, rejection_threshold, K_decision, averaging, when, CorruptValue, true_nodes, Print_Val, noise_STD, PrivacyMethod, p, learning_rate, batch_size, input_size, hidden_size, num_classes, detect, log_filename, test_ds, criterion, save, CorruptClients, typeAttack, var_attack, mean, Target, scale, num_clients, perm_threshold=0.5):
    # Initialize variables
    num_nodes = len(node_models)
    converged = False
    count = 0
    Error = []
    Track = 0
    mask_history = []
    loss_list = []
    acc_list = []
    TrainAcc = []
    TrainLoss = []
    # lying_nodes = lying_nodes or set()
    global_dict = node_models[0].state_dict() 
    True_avg_dict = global_dict
    # True_avg_dict = True_avg.state_dict()
    iters = [iter(DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)) for ds in client_datasets]

    # Detection 
    # mask = np.ones((num_nodes, num_nodes), dtype=int)  # 1: active, -1: blocked
    # Start with all zeros
    mask = np.zeros((num_nodes, num_nodes), dtype=int)

    for i, j in G.edges():
        mask[i, j] = 1
        mask[j, i] = 1

    D = {i: {j: 0 for j in G.neighbors(i)} for i in G.nodes()}  # Suspicion scores
    ignored = {i: set() for i in G.nodes()}  # Ignored neighbors

    # Model parameter initialization
    local_dicts = [model.state_dict() for model in node_models]

    # param_keys = True_avg.keys()  # all models have the same parameters
    A_ij = calc_incidence_nested(G)
    x_history = []
    # Initialize PDMM variables with tensor support
    x = [{} for _ in range(num_nodes)]
    z = [{} for _ in range(num_nodes)]
    y = [{} for _ in range(num_nodes)]
    y_transmit = [{} for _ in range(num_nodes)]
    # Initialize x with local models and move to device
    for i in range(num_nodes):
        for key in True_avg_dict.keys():
                x[i][key] = local_dicts[i][key].clone()

    # Initialize z and y
    for i in range(num_nodes):
        z[i] = {}
        y[i] = {}
        y_transmit[i] = {}
        for j in G.neighbors(i):
            z[i][j] = {}
            y[i][j] = {}
            y_transmit[i][j] = {}
            for key in True_avg_dict.keys():
                if PrivacyMethod == 3:
                    z[i][j][key] = torch.randn_like(True_avg_dict[key]) * noise_STD
                else:
                    z[i][j][key] = torch.zeros_like(True_avg_dict[key])
                y[i][j][key] = torch.zeros_like(True_avg_dict[key])
                y_transmit[i][j][key] = torch.zeros_like(True_avg_dict[key])
         
    if PrivacyMethod == 2:
        smpc_masks = {}
        for i in range(num_nodes):
            for j in G.neighbors(i):
                if i < j:
                    smpc_masks[(i, j)] = {}
                    for key in True_avg_dict:
                        smpc_masks[(i, j)][key] = torch.randn_like(True_avg_dict[key])




    honest_nodes  = set(true_nodes)
    lying_nodes   = [n for n in range(num_nodes) if n not in honest_nodes]

    edges = list(G.edges())
    edges_honest_target  = [(i, j) for (i, j) in edges if j in honest_nodes]
    edges_corrupt_target = [(i, j) for (i, j) in edges if j in lying_nodes]

    total_honest_edges   = len(edges_honest_target)
    total_corrupt_edges  = len(edges_corrupt_target)

    FAR_list, MDR_list = [], []      # will grow one element per PDMM iteration

    # print(lying_nodes)
    # Synchronous PDMM with detection
    while not converged and count < max_iters:
        g             = [{} for _ in range(num_nodes)]
        sum_loss      = 0.0          # Σ_i   loss_i
        sum_correct   = 0            # Σ_i   correct_i
        sum_samples   = 0            # Σ_i   N_i          (needed for accuracy)

        for i, model in enumerate(node_models):

            # ---- get next batch ---------------------------------------------------
            try:
                xb, yb = next(iters[i])
            except StopIteration:
                iters[i] = iter(DataLoader(client_datasets[i],
                                        batch_size=batch_size,
                                        shuffle=True, drop_last=True))
                xb, yb = next(iters[i])

            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

            # ---- forward & backward ----------------------------------------------
            with cuda_guard("forward+backward"):
                logits = model(xb)
                loss   = F.cross_entropy(logits, yb)

            model.zero_grad(set_to_none=True)
            loss.backward()
            for n, p in model.named_parameters():
                g[i][n] = p.grad.detach().clone()

            # ---- GLOBAL metrics accumulation -------------------------------------
            sum_loss    += loss.item() * yb.size(0)               # weight by batch size
            preds        = logits.argmax(dim=1)
            sum_correct += (preds == yb).sum().item()
            sum_samples += yb.size(0)

        # ------------- averaged metrics over ALL nodes ----------------------------
        Train_avg_loss = sum_loss / sum_samples
        Train_avg_acc  = 100.0 * sum_correct / sum_samples
        TrainLoss.append(Train_avg_loss)
        TrainAcc.append(Train_avg_acc)
        print(f"Iter {count:3d} | Train loss={Train_avg_loss:.4f} | Train acc={Train_avg_acc:.2f}%")
        torch.cuda.empty_cache() 

        #g = CorruptData(CorruptClients, g, typeAttack, var, mean, Target, num_nodes, scale)
        # for i in range(num_nodes):
        #     if CorruptClients[i]:
        #         total_norm = sum(torch.norm(g[i][k]) for k in g[i])
        #         print(f"Node {i}: grad norm = {total_norm:.2e}")

        # --------------------
        # 1. Synchronous x-update for all nodes
        # --------------------
        x_new = [{} for _ in range(num_nodes)]
        for i in range(num_nodes):
            # Count corrupt neighbors once for node i
            corrupt_neighbors = sum(1 for j in G.neighbors(i) if mask[i][j] == -1)
            effective_degree = G.degree[i] - corrupt_neighbors

            # Update each parameter using the same effective degree
            
            for key in True_avg_dict:
                num = local_dicts[i][key].clone() - learning_rate * g[i][key]
                for j in G.neighbors(i):
                    if mask[i][j] != -1:
                        num -= A_ij[i][j] * z[i][j][key]
                den = 1.0 +  c * effective_degree 
                x_new[i][key] = num / den
        x = x_new
        x = CorruptData(CorruptClients, x , typeAttack, var_attack, mean, Target, num_nodes, scale)

        x_history.append(x.copy())

        # ---- Sync torch modules and local_dicts with the fresh PDMM solution ----
        for i in range(num_nodes):
            # build a plain state-dict for loading
            new_state = {k: v.clone() for k, v in x[i].items()}
            node_models[i].load_state_dict(new_state)   # so next gradient is ?F(w_i^{(k)})
            local_dicts[i] = new_state                  # num = w_i^{(k)} - a g_i^{(k)}

            # —— Update the “true average” model so your consensus target moves ——
        for key in True_avg_dict:
        # average the parameter across all honest (or all) nodes
            True_avg_dict[key] = torch.stack([ x[i][key] for i in range(num_nodes) ]).mean(0)

        # ------------------------------------------------------------------------

        # --------------------
        # 2. Dual variable update (y)
        # --------------------
        for i in range(num_nodes):
            for j in G.neighbors(i):
                for key in True_avg_dict:
                    y[i][j][key] = z[i][j][key] + 2 * c * A_ij[i][j] * x[i][key]
                    if PrivacyMethod == 1:
                        y[i][j][key].add_(torch.randn_like(local_dicts[i][key]) * noise_STD)



        if PrivacyMethod == 2:
            # SMPC: Apply pairwise masks for secure aggregation
            for i in range(num_nodes):
                for j in G.neighbors(i):
                    for key in True_avg_dict:
                        if i < j:
                            y_transmit[i][j][key] = y[i][j][key] + smpc_masks[(i, j)][key]
                        else:
                            y_transmit[i][j][key] = y[i][j][key] - smpc_masks[(j, i)][key]
        else:
            # Original: no additional masking
            for i in range(num_nodes):
                for j in G.neighbors(i):
                    for key in True_avg_dict:
                         y_transmit[i][j][key] = y[i][j][key].clone()


        # --------------------
        # 3. Detection logic (executed periodically)
        # --------------------
        # if count > when:
        #print(f"Detecting at count {count}")
        if detect == True:
          for i in range(num_nodes):
              neighbors = [j for j in G.neighbors(i) if j not in ignored[i]]
              if not neighbors:
                  continue
  
              # Precompute absolute values of y variables (for PDMM minus-sign handling)
              abs_y = {j: {key: torch.abs(y_transmit[j][i][key]) for key in True_avg_dict} for j in neighbors}

              # 1. Compute element-wise median (m_i)
              medians = {}
              for key in True_avg_dict:
                  # Stack all neighbors' parameters for this key
                  params = torch.stack([abs_y[j][key] for j in neighbors])
                  medians[key] = torch.median(params, dim=0).values  # Element-wise median

              # 2. Compute Delta Y_{i,j} using infinity norm
              delta_ys = []
              for j in neighbors:
                  max_diff = -float('inf')
                  for key in True_avg_dict:
                      clean_a = torch.nan_to_num(abs_y[j][key], nan=0.0, posinf=1e6, neginf=-1e6)
                      clean_b = torch.nan_to_num(medians[key],   nan=0.0, posinf=1e6, neginf=-1e6)
                      diff = torch.max(torch.abs(clean_a - clean_b)).item()


                      # Safely convert to float
                      if isinstance(diff, torch.Tensor):
                          diff = diff.item()

                      # Handle NaN/inf by treating it as a detection
                      if not np.isfinite(diff):
                          diff = 1e9  # Forces detection
                      if diff > max_diff:
                          max_diff = diff
                  delta_ys.append(max_diff)

              # 3. Compute MAD and threshold
              median_delta = np.median(delta_ys)
              deviations = np.abs(delta_ys - median_delta)
              MAD_val = np.median(deviations)
              threshold = rejection_threshold * MAD_val
              epsilon = 1e-12  
              threshold = max(threshold, epsilon) # TO avaoid zero threshold

              # 4. Update suspicion scores
              for idx, j in enumerate(neighbors):
                #   if j in lying_nodes:
                  # print(f"Node {i} check {j}: ?Y={delta_ys[idx]:.2f}, threshold={threshold:.2f}")
                  if delta_ys[idx] > 0 and j in lying_nodes:
                #   if j in lying_nodes:
                      D[i][j] += 1
                      print(f"To {i} Value of {j}: ?Y={delta_ys[idx]:.8f}, threshold={threshold:.8f}, D = {D[i][j]}")
                  # if j in lying_nodes:
                      # if Print_Val:
                      # print(f"Node {i} suspicious of {j}: ?Y={delta_ys[idx]:.2f}, threshold={threshold:.2f}, D = {D[i][j]}")

              # 5. Periodic mitigation check
              if count % K_decision == 0 and count > 0:
                  for j in list(D[i].keys()):  # Iterate over copy to allow modification
                      if D[i][j] > K_decision/2:
                          # print(f"Node {i} ignoring node {j} for next {K_decision} iterations")
                          mask[i][j] = -1
                      else:
                          mask[i][j] = 1
                      D[i][j] = 0
        mask_history.append(mask.copy())

        false_alarms       = sum(1 for (i, j) in edges_honest_target  if mask[i, j] == -1)
        missed_detections  = sum(1 for (i, j) in edges_corrupt_target if mask[i, j] != -1)

        far = false_alarms      / total_honest_edges  if total_honest_edges  else 0.0
        mdr = missed_detections / total_corrupt_edges if total_corrupt_edges else 0.0

        FAR_list.append(far)
        MDR_list.append(mdr)

        # --------------------
        # 4. Synchronous z-update with masking
        # --------------------

        if PrivacyMethod == 2:
            for i in range(num_nodes):
                for j in G.neighbors(i):
                    if mask[i][j] == -1:
                        for key in True_avg_dict:
                            z[i][j][key] = (1 - averaging) * z[i][j][key] 
                    if mask[i][j] == 1:
                        for key in True_avg_dict:
                            if j < i:
                                unmasked = y_transmit[j][i][key] + smpc_masks[(j, i)][key]
                            else:
                                unmasked = y_transmit[j][i][key] - smpc_masks[(i, j)][key]
                            z[i][j][key] = (1 - averaging) * z[i][j][key] + (averaging) * unmasked
        else:                        
          for i in range(num_nodes):
              for j in G.neighbors(i):
                if mask[i][j] == -1:
                      # Apply noise to blocked channels
                    for key in True_avg_dict:
                        z[i][j][key] = (1 - averaging) * z[i][j][key] 
                if mask[i][j] == 1:
                    # Normal update from y values
                    for key in True_avg_dict:
                        z[i][j][key] = (1 - averaging) * z[i][j][key]  + (averaging) * y_transmit[j][i][key].clone()


        # --------------------
        # 5. Update global model and check convergence
        # --------------------
        avg_error = 0
        err_i = 0
        total_elements = 0

        for i in true_nodes:
            node_error = 0
            for key in True_avg_dict:
                diff = x[i][key] - True_avg_dict[key]
                norm_diff = torch.norm(diff).item()**2
                node_error += norm_diff
            err_i += node_error
            avg_error = err_i**0.5
            total_elements += 1

        # Final norm: average error divided by the total number of nodes
        avg_error /= total_elements

        # Store the computed average error
        Error.append(avg_error)
        # if count % 100 == 0:
        #     loss1, acc = evaluate(node_models, test_ds, criterion, batch_size)
        #     print(f"Iter {count:3d} | FAR={far:.3f} | MDR={mdr:.3f} | loss={loss:.4f} | acc={acc:.2f}%")
        #     loss_list.append(loss)
        #     acc_list.append(acc) 
        #     print(f"cunt {count} and error {Error[-1]}" )

        if avg_error < tolerance:
            print(f'Converged at iteration {count}')
            converged = True
        elif count % 10 == 0 and Print_Val:
            print(f'Iter {count}: Error {avg_error:.4f}')

        # ─── inside the while-loop, right after you update FAR/MDR/Error ─────────────
        # adaptive evaluation frequency: 0–50 iters → every 5; 50–199 → 10; 200+ → 20
        if   count < 50:      eval_every = 5
        elif count < 200:     eval_every = 10
        else:                 eval_every = 20

        if count % eval_every == 0:
            honest_models = [node_models[i] for i in true_nodes]
            loss1, acc = evaluate(honest_models, test_ds, criterion, batch_size)
            # loss1,acc = 0,0
            print(f"Iter {count:3d} | FAR={far:.3f} | MDR={mdr:.3f} | "
                f"loss={loss1:.4f} | acc={acc:.2f}%")
            loss_list.append(loss1)
            acc_list.append(acc)

        # ─── logging (only once per evaluation) ──────────────────────────────────────
        if save and count % eval_every == 0:
            with open(log_filename, "a") as logfile:
                logfile.write(
                    f"Iter {count:3d}  FAR={far:.3f} | MDR={mdr:.3f} | "
                    f"loss={loss1:.4f} | acc={acc:.2f}%\n"
                )

        count += 1
        # ─────────────────────────────────────────────────────────────────────────────


        # if save == True:
        #     with open(log_filename, "a") as logfile:
        #         logfile.write(f"Round {count}  FAR={far:.3f} | MDR={mdr:.3f} | loss={loss1:.4f} | acc={acc:.2f}%")
        # count += 1


    # Update final models
    for i in range(num_nodes):
        model_dict = node_models[i].state_dict()
        for key in True_avg_dict:
            model_dict[key] = x[i][key].clone()
        node_models[i].load_state_dict(model_dict)


            

    # Create an averaged model with the final True_avg_dict
    averaged_model = type(node_models[0])(input_size, hidden_size, num_classes).to(device)  # Pass necessary arguments directly
    averaged_model.load_state_dict(True_avg_dict)
    

    return node_models, Error, FAR_list, MDR_list, mask_history, loss_list, acc_list, TrainAcc, TrainLoss

    
def evaluate(models, test_dataset, criterion, batch_size):
    """Evaluate one or multiple models on the test set."""
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    if not isinstance(models, (list, tuple)):
        models = [models]
    total_loss, total_acc = 0.0, 0.0
    for model in models:
        model.to(device).eval()
        loss_sum, correct, count = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss_sum += criterion(outputs, labels).item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                count += labels.size(0)
        total_loss += loss_sum / len(loader)
        total_acc += 100 * correct / count
    return total_loss / len(models), total_acc / len(models)
    
    
def main():
    # Reproducibility
    Seed = 42  # for reproducibility
    random.seed(Seed)
    np.random.seed(Seed)
    torch.manual_seed(Seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Seed)
        torch.cuda.manual_seed_all(Seed)

    # Graph setup
    required_probability = 0.9999
    Amount_Clients = 20
    num_nodes, G, A, pos, r_c = build_random_graph(
        Amount_Clients, required_probability, fix_num_nodes=True
    )
    print("num_nodes:", num_nodes)

    # Hyperparameters
    input_size = 28 * 28        # MNIST images are 28x28 pixels
    hidden_size = 128           # Number of neurons in the hidden layer
    num_classes = 10            # Number of output classes (digits 0-9)
    # num_epochs = 5              # Number of local training epochs per aggregation
    batch_size = 64             # Batch size for training
    learning_rate = 0.01        # Learning rate for the optimizer
    num_clients = num_nodes     # Number of clients
    num_rounds = 1              # Number of aggregation rounds
    # threshold = 0.02            # Loss threshold for stopping criteria

    # Corruption settings
    typeAttackArray = [6,1,3,4,5]              # 0: No attack, 1: Gaussian noise, 2: Copycat attack, 3: Gaussian addative noise attack, 4: LIE attack, 5 Sign flip attack, 6: Label Flipping attack
    typeAttacknode = typeAttackArray[0]  
    if typeAttacknode == 0:
        percentageCorrupt = 0 /num_nodes
    else:
        percentageCorrupt = 2 /num_nodes
    
    corrupt = True
    detect = True
    save = True
    CorruptClients = CorruptGeneration(
        percentageCorrupt, corrupt, num_clients
    )
    lying_nodes = np.where(CorruptClients == 1)[0]
    true_nodes = [i for i in range(num_nodes) if i not in lying_nodes]
    print("Corrupt Clients:", lying_nodes)
    # print(num_clients)
    tolerance = -1               # PDMM tolerance
    c = 0.5                      # PDMM c
    max_iters =  250           # PDMM max iterations

    when = 0
    CorruptValue = -1e17
    rejection_threshold = 6
    K_decision = 5
    averaging = 0.5
    # noise_levels = [0,10**4]  # Noise levels for Gaussian noise
    noise_levels = [0,10**1,10**-1,10**0,10**2]  # Noise levels for Gaussian noise



    # noise_levels = [10**0]  # Noise levels for Gaussian noise

    var_attack = 10**3
    mean = 0
    Target = np.random.randint(1, num_clients)
    scale = 1
    PrivacyMethodArray = [3,1,2] # 0: No privacy, 1: DP, 2: SMPC, 3: Subspace
    p = 0
    PrimModulo = [0, 2**61 - 1] # Prime number for SMPC should be above sum of all client local model values
    print(CorruptClients)
    for typeAttack in typeAttackArray:
      for PrivacyMethod in PrivacyMethodArray:
            
        # Visualize graph
        neighbors_dict = {ln: list(G.neighbors(ln)) for ln in lying_nodes}
        print("Neighbors of lying nodes:", neighbors_dict)
        plt.figure(figsize=(8,6))
        color_map = ['red' if n in lying_nodes else 'blue' for n in G.nodes()]
        nx.draw(G, pos, with_labels=True, node_color=color_map)
        plt.title("Graph with Lying (Red) and Honest (Blue) Nodes")
        plt.show()

        # Honest-only connectivity checks
        remaining_nodes = [n for n in G.nodes() if n not in lying_nodes]
        G_sub = G.subgraph(remaining_nodes)
        still_connected = nx.is_connected(G_sub)
        print("Are the honest-only nodes still forming a connected subgraph?", still_connected)

        all_good = True
        for node in remaining_nodes:
            neighbors = list(G.neighbors(node))
            if neighbors:
                honest_neighbors = sum(1 for nb in neighbors if nb not in lying_nodes)
                if honest_neighbors <= len(neighbors)/2:
                    print(f"Honest node {node} has only {honest_neighbors}/{len(neighbors)} honest neighbors")
                    all_good = False
        print("All honest nodes have majority honest neighbors?", all_good)

        # Data loading
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_ds = torchvision.datasets.MNIST(
            root='./data', train=True, transform=transform, download=True
        )
        test_ds = torchvision.datasets.MNIST(
            root='./data', train=False, transform=transform, download=True
        )
        client_datasets = split_dataset(train_ds, num_clients)
    
        # Optional label flipping
        for idx in lying_nodes:
            if typeAttack == 6:
                client_datasets[idx] = FlipLabelDataset(client_datasets[idx])

        # Training/aggregation loop
        criterion = nn.CrossEntropyLoss()
        results = []
        if PrivacyMethod == 2:
            loop_params = PrimModulo
            param_name  = 'p'
        else:
            loop_params = noise_levels
            param_name  = 'noise_STD'
        for param in loop_params:
            if save == True:
                log_filename = f"results_PDMM_Attack{typeAttack}_Privacy{PrivacyMethod}_{param_name}={param} .txt"
            for rnd in range(num_rounds):

                      # Initialize models
                init_w = Current(input_size, hidden_size, num_classes).state_dict()
                local_models = []
                for _ in range(num_clients):
                    m = Current(input_size, hidden_size, num_classes).to(device)
                    m.load_state_dict(init_w)
                    m.train()
                    local_models.append(m)
                # Apply corruption
                # local_models = CorruptData(
                #     CorruptClients, local_models,
                #     typeAttack, var, mean, Target,
                #     num_clients, scale
                # )
                # Aggregate via PDMM
                print(num_clients, "clients with gradients 3")
                updated, err_hist, FAR_list, MDR_list, mask_history, loss_list, acc_list, TrainAcc, TrainLoss = aggregate_models(
                        client_datasets, local_models, G,
                        tolerance, c, max_iters, rejection_threshold,
                        K_decision, averaging, when, CorruptValue,
                        true_nodes, False, (0 if PrivacyMethod==2 else param), PrivacyMethod,
                        (param if PrivacyMethod==2 else 0), learning_rate, batch_size, input_size, hidden_size,
                        num_classes, detect, log_filename, test_ds, criterion, save, 
                        CorruptClients, typeAttack, var_attack, mean, Target, num_clients, scale)

            result_dict = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "privacy":   PrivacyMethod,
                    "attack":    typeAttack,
                    param_name:  param,
                    "FAR":       FAR_list,
                    "MDR":       MDR_list,
                    "Error":     err_hist,
                    "test_loss":      loss_list,
                    "test_accuracy":  acc_list,
                    "train_loss":      TrainLoss,
                    "train_accuracy":  TrainAcc,
            }

            # 1) keep it in-memory if you still want the combined figure
            results.append(result_dict)

            # 2) write it permanently right now
            append_result(result_dict)          # ← NEW, one line

if __name__ == "__main__":
    main()

