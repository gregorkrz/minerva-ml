python -m src.jobs.train -name initial_training_long_l1 --run
python -m src.jobs.train -name initial_training_long_l1_pretrained --run --use-pretrained pretrain_s


python -m src.jobs.train -name initial_traing_longl1 --run --loss l1
python -m src.jobs.train -name initial_traing_longl1 --run --loss l1 --use-pretrained pretrain_s


python -m src.jobs.train -name initial_traing_long_huber_pretrained --run --loss huber --use-pretrained pretrain_s
python -m src.jobs.train -name initial_traing_long_huber --run --loss huber 

