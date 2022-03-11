srun -w ampere2 --cpus-per-task 50 --mem-per-cpu 1000M \
    python -m core.gdrn_selfocc_modeling.tools.denstereo.generate_pbr_P \
        --dataset denstereo --split train_pbr_left --threads 50 &
srun -w ampere2 --cpus-per-task 50 --mem-per-cpu 1000M \
    python -m core.gdrn_selfocc_modeling.tools.denstereo.generate_pbr_P \
        --dataset denstereo --split train_pbr_right --threads 50
