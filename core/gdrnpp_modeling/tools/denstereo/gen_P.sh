DATASET=denstereo
THREADS=50
srun -w ampere2 --cpus-per-task $THREADS --mem-per-cpu 1000M \
    python -m core.gdrnpp_modeling.tools.$DATASET.generate_pbr_P \
        --dataset $DATASET --split train_pbr_left --threads $THREADS &
srun -w ampere2 --cpus-per-task $THREADS --mem-per-cpu 1000M \
    python -m core.gdrnpp_modeling.tools.$DATASET.generate_pbr_P \
        --dataset $DATASET --split train_pbr_right --threads $THREADS
