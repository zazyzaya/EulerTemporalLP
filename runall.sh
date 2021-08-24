python run.py --dataset OPTC -d 0.5 -t 5 -r LSTM
python run.py --dataset OPTC -d 1.0 -t 5 -r LSTM
python run.py --dataset OPTC -d 2.0 -t 5 -r LSTM
python run.py --dataset OPTC -d 4.0 -t 5 -r LSTM
python run.py --dataset OPTC -d 8.0 -t 5 -r LSTM
python run.py --dataset OPTC -d 12.0 -t 5 -r LSTM
mv results/stats.txt results/OpTC_static.txt
python run.py --dataset OPTC -d 0.5 -t 5 -p -r LSTM
python run.py --dataset OPTC -d 1.0 -t 5 -p -r LSTM
python run.py --dataset OPTC -d 2.0 -t 5 -p -r LSTM
python run.py --dataset OPTC -d 4.0 -t 5 -p -r LSTM
python run.py --dataset OPTC -d 8.0 -t 5 -p -r LSTM
python run.py --dataset OPTC -d 12.0 -t 5 -p -r LSTM
mv results/stats.txt results/OpTC_dynamic.txt