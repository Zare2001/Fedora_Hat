try:
    import pickle5 as pickle
    print("?? using pickle5 (protocol 5 support)")
except ImportError:
    import pickle
    print("?? using stdlib pickle (no protocol 5 support)")


# 1. Load all entries
results = []
with open('resultsFedAVG.pkl', 'rb') as f:
    while True:
        try:
            results.append(pickle.load(f))
        except EOFError:
            break

# 2. Print a summary line for each entry
for i, res in enumerate(results):
    attack  = res.get('attack')
    privacy = res.get('privacy')
    # depending on privacy, the hyper-param key is either 'noise_STD' or 'p'
    param_name = 'p' if privacy == 2 else 'noise_STD'
    param_val  = res.get(param_name, None)
    print(f"Entry {i:2d} ? attack={attack}, privacy={privacy}, {param_name}={param_val}")
