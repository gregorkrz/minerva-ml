template = "python -m src.scripts.train -bs 2048 --mode regression -E-available-no-muon -name Run_cond_only_full_seed42 --d_model 128 --mlp_layers {nlayers} --dropout 0.0 --cond_only --seed {seed} -seed-event-sampler 42 --max_steps 500000 --grad_accum_steps 1 --lr 1e-3"
for seed in [42]:
    for layers in [4]:
        print(template.format(seed=seed, nlayers=layers))

