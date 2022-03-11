srun -w ampere2 --cpus-per-task 55 --mem-per-cpu 1500M \
    python -m core.gdrn_selfocc_modeling.tools.denstereo.generate_pbr_P \
        --dataset denstereo --split train_pbr_left --threads 50 &
srun -w ampere2 --cpus-per-task 55 --mem-per-cpu 1500M \
    python -m core.gdrn_selfocc_modeling.tools.denstereo.generate_pbr_P \
        --dataset denstereo --split train_pbr_right --threads 50
