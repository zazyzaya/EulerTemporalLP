python run.py -t 5 -i TEdge -d 0.5
python run.py -t 5 -i TEdge -d 1
python run.py -t 5 -i TEdge -d 1.5
python run.py -t 5 -i TEdge -d 2
python run.py -t 5 -i TEdge -d 2.5
python run.py -t 5 -i TEdge -d 3
mv results/stats.txt results/LANL_GRU.txt
python run.py -t 5 -i TEdge -d 0.5 -r LSTM
python run.py -t 5 -i TEdge -d 1 -r LSTM
python run.py -t 5 -i TEdge -d 1.5 -r LSTM
python run.py -t 5 -i TEdge -d 2 -r LSTM
python run.py -t 5 -i TEdge -d 2.5 -r LSTM
python run.py -t 5 -i TEdge -d 3 -r LSTM
mv results/stats.txt results/LANL_LSTM.txt
python run.py -t 5 -i DynTEdge -d 0.5
python run.py -t 5 -i DynTEdge -d 1
python run.py -t 5 -i DynTEdge -d 1.5
python run.py -t 5 -i DynTEdge -d 2
python run.py -t 5 -i DynTEdge -d 2.5
python run.py -t 5 -i DynTEdge -d 3
mv results/stats.txt results/LANL_DynSM_GRU.txt
python run.py -t 5 -i DynTEdge -d 0.5 -r LSTM
python run.py -t 5 -i DynTEdge -d 1 -r LSTM
python run.py -t 5 -i DynTEdge -d 1.5 -r LSTM
python run.py -t 5 -i DynTEdge -d 2 -r LSTM
python run.py -t 5 -i DynTEdge -d 2.5 -r LSTM
python run.py -t 5 -i DynTEdge -d 3 -r LSTM
mv results/stats.txt results/LANL_DynSM_LSTM.txt
python run.py -t 5 -i Dynamic -d 0.125 --dataset OpTC
python run.py -t 5 -i Dynamic -d 0.25 --dataset OpTC
python run.py -t 5 -i Dynamic -d 0.5 --dataset OpTC
python run.py -t 5 -i Dynamic -d 1 --dataset OpTC
python run.py -t 5 -i Dynamic -d 1.5 --dataset OpTC
python run.py -t 5 -i Dynamic -d 2 --dataset OpTC
python run.py -t 5 -i Dynamic -d 2.5 --dataset OpTC
python run.py -t 5 -i Dynamic -d 3 --dataset OpTC
mv results/stats.txt results/OpTC_dyn_GRU.txt 
python run.py -t 5 -i Dynamic -d 0.125 --dataset OpTC -r LSTM
python run.py -t 5 -i Dynamic -d 0.25 --dataset OpTC -r LSTM
python run.py -t 5 -i Dynamic -d 0.5 --dataset OpTC -r LSTM
python run.py -t 5 -i Dynamic -d 1 --dataset OpTC -r LSTM
python run.py -t 5 -i Dynamic -d 1.5 --dataset OpTC -r LSTM
python run.py -t 5 -i Dynamic -d 2 --dataset OpTC -r LSTM
python run.py -t 5 -i Dynamic -d 2.5 --dataset OpTC -r LSTM
python run.py -t 5 -i Dynamic -d 3 --dataset OpTC -r LSTM
mv results/stats.txt results/OpTC_dyn_LSTM.txt 