python run.py -t 5 -i TEdge --dataset OpTC -d 0.125
python run.py -t 5 -i TEdge --dataset OpTC -d 0.25
python run.py -t 5 -i TEdge --dataset OpTC -d 0.5
python run.py -t 5 -i TEdge --dataset OpTC -d 1
python run.py -t 5 -i TEdge --dataset OpTC -d 1.5
python run.py -t 5 -i TEdge --dataset OpTC -d 2
python run.py -t 5 -i TEdge --dataset OpTC -d 2.5
python run.py -t 5 -i TEdge --dataset OpTC -d 3
mv results/stats.txt results/OpTC_GRU.txt
python run.py -t 5 -i TEdge --dataset OpTC -d 0.125 -r LSTM
python run.py -t 5 -i TEdge --dataset OpTC -d 0.25 -r LSTM
python run.py -t 5 -i TEdge --dataset OpTC -d 0.5 -r LSTM
python run.py -t 5 -i TEdge --dataset OpTC -d 1 -r LSTM
python run.py -t 5 -i TEdge --dataset OpTC -d 1.5 -r LSTM
python run.py -t 5 -i TEdge --dataset OpTC -d 2 -r LSTM
python run.py -t 5 -i TEdge --dataset OpTC -d 2.5 -r LSTM
python run.py -t 5 -i TEdge --dataset OpTC -d 3 -r LSTM
mv results/stats.txt results/OpTC_LSTM.txt
python run.py -t 5 -i DynTEdge --dataset OpTC -d 0.125
python run.py -t 5 -i DynTEdge --dataset OpTC -d 0.25
python run.py -t 5 -i DynTEdge --dataset OpTC -d 0.5
python run.py -t 5 -i DynTEdge --dataset OpTC -d 1
python run.py -t 5 -i DynTEdge --dataset OpTC -d 1.5
python run.py -t 5 -i DynTEdge --dataset OpTC -d 2
python run.py -t 5 -i DynTEdge --dataset OpTC -d 2.5
python run.py -t 5 -i DynTEdge --dataset OpTC -d 3
mv results/stats.txt results/OpTC_DynSM_GRU.txt
python run.py -t 5 -i DynTEdge --dataset OpTC -d 0.125 -r LSTM
python run.py -t 5 -i DynTEdge --dataset OpTC -d 0.25 -r LSTM
python run.py -t 5 -i DynTEdge --dataset OpTC -d 0.5 -r LSTM
python run.py -t 5 -i DynTEdge --dataset OpTC -d 1 -r LSTM
python run.py -t 5 -i DynTEdge --dataset OpTC -d 1.5 -r LSTM
python run.py -t 5 -i DynTEdge --dataset OpTC -d 2 -r LSTM
python run.py -t 5 -i DynTEdge --dataset OpTC -d 2.5 -r LSTM
python run.py -t 5 -i DynTEdge --dataset OpTC -d 3 -r LSTM
mv results/stats.txt results/OpTC_DynSM_LSTM.txt
python run.py -t 5 -i TEdge -d 0.125
python run.py -t 5 -i TEdge -d 0.25
python run.py -t 5 -i TEdge -d 0.5
python run.py -t 5 -i TEdge -d 1
python run.py -t 5 -i TEdge -d 1.5
python run.py -t 5 -i TEdge -d 2
python run.py -t 5 -i TEdge -d 2.5
python run.py -t 5 -i TEdge -d 3
mv results/stats.txt results/LANL_GRU.txt
python run.py -t 5 -i TEdge -d 0.125 -r LSTM
python run.py -t 5 -i TEdge -d 0.25 -r LSTM
python run.py -t 5 -i TEdge -d 0.5 -r LSTM
python run.py -t 5 -i TEdge -d 1 -r LSTM
python run.py -t 5 -i TEdge -d 1.5 -r LSTM
python run.py -t 5 -i TEdge -d 2 -r LSTM
python run.py -t 5 -i TEdge -d 2.5 -r LSTM
python run.py -t 5 -i TEdge -d 3 -r LSTM
mv results/stats.txt results/LANL_LSTM.txt
python run.py -t 5 -i DynTEdge -d 0.125
python run.py -t 5 -i DynTEdge -d 0.25
python run.py -t 5 -i DynTEdge -d 0.5
python run.py -t 5 -i DynTEdge -d 1
python run.py -t 5 -i DynTEdge -d 1.5
python run.py -t 5 -i DynTEdge -d 2
python run.py -t 5 -i DynTEdge -d 2.5
python run.py -t 5 -i DynTEdge -d 3
mv results/stats.txt results/LANL_DynSM_GRU.txt
python run.py -t 5 -i DynTEdge -d 0.125 -r LSTM
python run.py -t 5 -i DynTEdge -d 0.25 -r LSTM
python run.py -t 5 -i DynTEdge -d 0.5 -r LSTM
python run.py -t 5 -i DynTEdge -d 1 -r LSTM
python run.py -t 5 -i DynTEdge -d 1.5 -r LSTM
python run.py -t 5 -i DynTEdge -d 2 -r LSTM
python run.py -t 5 -i DynTEdge -d 2.5 -r LSTM
python run.py -t 5 -i DynTEdge -d 3 -r LSTM
mv results/stats.txt results/LANL_DynSM_LSTM.txt