#%%
import itertools

"""TABSYN"""
# # epochs1_vals = [2000, 5000]
# denoising_dim = [512, 1024]
# # factor_vals = [32, 64]
# lr1_vals = [ "1e-3", "1e-4"]# 
# lr2_vals = [ "1e-3", "1e-4"] #"1e-3",
# batch_size1 = [4096, 2048]
# batch_size2 = [4096, 2048]
# # beta = [1.0]#,0.1,0.01,0.001,0.0001, 0.00001]
# combinations = list(itertools.product(denoising_dim, lr1_vals, lr2_vals,batch_size1, batch_size2))

# for comb in combinations:
#     latent_dim, lr1, lr2, batch_size1, batch_size2 = comb
#     cmd = f"'--denoising_dim {latent_dim} --lr1 {lr1} --lr2 {lr2} --batch_size1 {batch_size1} --batch_size2 {batch_size2}'" #--factor {factor}
#     print(cmd)

# """STASY"""
# sigma_min = [0.01, 0.1]
# sigma_max = [5.0, 10.0]
# lr1_vals = [ "2e-3", "2e-4"]# 
# beta0 = [0.80, 0.90, 0.95]
# alpha0 = [0.20, 0.25, 0.30]
# # beta = [1.0]#,0.1,0.01,0.001,0.0001, 0.00001]
# combinations = list(itertools.product(sigma_min,sigma_max,lr1_vals,beta0,alpha0))
# for comb in combinations:
#     sig_min, sig_max, lr1, b0, a0 = comb
#     cmd = f"'--model.sigma_min {sig_min} --model.sigma_max {sig_max} --optim.lr {lr1} --model.beta0 {b0} --model.alpha0 {a0}'"
#     print(cmd)

"""TabDDPM"""
dim_embed = [128, 256, 512, 1024]
lr1_vals = [ "1e-3", "2e-3","2e-4"]# 
num_layers = [2,4]
batch_size = [2048, 4096]
# beta = [1.0]#,0.1,0.01,0.001,0.0001, 0.00001]
combinations = list(itertools.product(dim_embed, lr1_vals, num_layers, batch_size))
for comb in combinations:
    emb, lr1, nl, bs = comb
    cmd = f"'--lr {lr1} --dim_embed {emb} --batch_size {bs} --num_layers {nl}'"
    print(cmd)
