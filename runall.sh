python run.py -t 5 -d 0.25 -i dynamic
python run.py -t 5 -d 0.5 -i dynamic
python run.py -t 5 -d 1 -i dynamic
python run.py -t 5 -d 2 -i dynamic 
python run.py -t 5 -d 3 -i dynamic 
python run.py -t 5 -d 4 -i dynamic 
mv results/stats.txt results/LANL_dyn_gru.txt
python run.py -t 5 -d 0.25 -i dynamic -r LSTM
python run.py -t 5 -d 0.5 -i dynamic -r LSTM
python run.py -t 5 -d 1 -i dynamic -r LSTM
python run.py -t 5 -d 2 -i dynamic -r LSTM
python run.py -t 5 -d 3 -i dynamic -r LSTM 
python run.py -t 5 -d 4 -i dynamic -r LSTM
mv results/stats.txt results/LANL_dyn_lstm.txt
python run.py -t 5 -d 0.5 -i dyntedge
python run.py -t 5 -d 1 -i dyntedge
python run.py -t 5 -d 2 -i dyntedge
python run.py -t 5 -d 3 -i dyntedge
python run.py -t 5 -d 4 -i dyntedge
mv results/stats.txt results/LANL_dyn_tedge_gru.txt
python run.py -t 5 -d 0.5 -i dyntedge -r LSTM
python run.py -t 5 -d 1 -i dyntedge -r LSTM
python run.py -t 5 -d 2 -i dyntedge -r LSTM
python run.py -t 5 -d 3 -i dyntedge -r LSTM 
python run.py -t 5 -d 4 -i dyntedge -r LSTM
mv results/stats.txt results/LANL_dyn_tedge_lstm.txt
python run.py -t 5 -d 0.125 -i dynamic --dataset OpTC
python run.py -t 5 -d 0.25 -i dynamic --dataset OpTC
python run.py -t 5 -d 0.5 -i dynamic --dataset OpTC
python run.py -t 5 -d 1 -i dynamic --dataset OpTC
python run.py -t 5 -d 1.5 -i dynamic --dataset OpTC
python run.py -t 5 -d 2 -i dynamic --dataset OpTC
python run.py -t 5 -d 2.5 -i dynamic --dataset OpTC
python run.py -t 5 -d 3 -i dynamic --dataset OpTC 
python run.py -t 5 -d 4 -i dynamic --dataset OpTC 
mv results/stats.txt results/OpTC_dyn_gru.txt
python run.py -t 5 -d 0.125 -i dynamic --dataset OpTC -r LSTM
python run.py -t 5 -d 0.25 -i dynamic --dataset OpTC -r LSTM
python run.py -t 5 -d 0.5 -i dynamic --dataset OpTC -r LSTM
python run.py -t 5 -d 1 -i dynamic --dataset OpTC -r LSTM
python run.py -t 5 -d 1.5 -i dynamic --dataset OpTC -r LSTM
python run.py -t 5 -d 2 -i dynamic --dataset OpTC -r LSTM
python run.py -t 5 -d 2.5 -i dynamic --dataset OpTC -r LSTM
python run.py -t 5 -d 3 -i dynamic --dataset OpTC -r LSTM
python run.py -t 5 -d 4 -i dynamic --dataset OpTC -r LSTM 
mv results/stats.txt results/OpTC_dyn_lstm.txt
python run.py -t 5 -d 3 -i dyntedge
python run.py -t 5 -d 4 -i dyntedge
mv results/stats.txt results/LANL_dyn_gru_cont.txt 
python run.py -t 5 -d 3 -i dyntedge -r LSTM 
python run.py -t 5 -d 4 -i dyntedge -r LSTM 
mv results/stats.txt results/LANL_dyn_lstm_cont.txt 