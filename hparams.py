import os

# for lr in [0.0005, 0.0008, 0.001, 0.003]:
#     for batch in [24]:
#         for network in ["_Stan"]:#,
#                         # "_drop_Stan_normal_new"]: #, "_activated_Stan", "_maxpool_Stan", "_activated_maxpool_Stan"]: #, "_50_alphaweights_Stan"]: #
#             for data in ["_normal_new"]:
#                 for drop in [0.15, 0.20, 0.25]:
#                     os.system(f"python3 train_CADET.py {lr} {batch} {network} {data} {drop}")

for lr in [0.0016]: #, 0.001, 0.0006]: #0.0006, 0.0008, 0.001]:
    for batch in [24]:
        for network in [ "_relu_Stan"]: #, "_relu_Stan"]: #, "_50_prelu_Stan"]: #, "_relu_Stan", "_Stan"]:#,
                        # "_drop_Stan_normal_new"]: #, "_activated_Stan", "_maxpool_Stan", "_activated_maxpool_Stan"]: #, "_50_alphaweights_Stan"]: #
            for cavities in ["_100"]: #"_50", "_90","_100"]:
                for data in ["_normal_final"]:
                    for drop in [0.0]: #, 0.2]: #, 0.0]: # 0.15, 0.0]:  #[0.0, 0.15, 0.3]:
                        os.system(f"python3 train_CADET.py {lr} {batch} {network+cavities} {data} {drop}")