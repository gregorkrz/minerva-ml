# Regression: available energy (no muon), cond-only MLP
template_regression = "python -m src.scripts.train -bs 2048 --mode regression -E-available-no-muon -name Run_cond_only_lowLR_NR_full_seed{seed} --d_model 128 --mlp_layers {nlayers} --dropout 0.0 --cond_only --seed {seed} -seed-event-sampler 42 --max_steps 1000000 --grad_accum_steps 1 --lr 1e-4 --data_path /global/cfs/cdirs/m3246/gregork/Minerva/20260326 --zero-cond-feature 2"
template_classifier = "python -m src.scripts.train -bs 2048 --mode classifier -npi2 -name Run_cond_only_lowLR_classifier_NR_full_seed{seed} --d_model 128 --mlp_layers {nlayers} --dropout 0.0 --cond_only --seed {seed} -seed-event-sampler 42 --max_steps 1000000 --grad_accum_steps 1 --lr 1e-4 --data_path /global/cfs/cdirs/m3246/gregork/Minerva/20260326 --zero-cond-feature 2"


for seed in [50, 51, 52, 53]:
    for layers in [4]:
        print(template_regression.format(seed=seed, nlayers=layers))


for seed in [50, 51, 52, 53]:
    for layers in [4]:
        print(template_classifier.format(seed=seed, nlayers=layers))

# print also the same, but with dataset cap
# templtate_with_dscap = "python -m src.scripts.train -bs 2048 --mode regression -E-available-no-muon -name Run_cond_only_{dscap}_seed{seed} --d_model 128 --mlp_layers {nlayers} --dropout 0.0 --cond_only --seed {seed} -seed-event-sampler 42 --max_steps 100000 --grad_accum_steps 1 --lr 1e-3 -cap {cap}"
# for seed in [43, 44, 45, 46]:
#    for layers in [4]:
#        for cap in [20000, 50000, 100000]:
#            print(templtate_with_dscap.format(seed=seed, nlayers=layers, cap=cap, dscap=cap))
