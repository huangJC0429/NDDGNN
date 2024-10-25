python -m src.run --dataset cora_ml --use_best_hyperparams --num_runs 10
python -m src.run --dataset citeseer_full --use_best_hyperparams --num_runs 10

python -m src_large.run --dataset chameleon --use_best_hyperparams --num_runs 10
python -m src_large.run --dataset squirrel --use_best_hyperparams --num_runs 10
python -m src_large.run --dataset directed-roman-empire --use_best_hyperparams --num_runs 10
python -m src_large.run --dataset arxiv-year --use_best_hyperparams --num_runs 10
python -m src_large.run --dataset snap-patents --use_best_hyperparams --num_runs 10