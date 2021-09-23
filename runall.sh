python run.py --dataset OpTC -t 5 -d 0.125 -i DynTEdge
python run.py --dataset OpTC -t 5 -d 0.25 -i DynTEdge
python run.py --dataset OpTC -t 5 -d 0.5 -i DynTEdge
python run.py --dataset OpTC -t 5 -d 1 -i DynTEdge
python run.py --dataset OpTC -t 5 -d 1.5 -i DynTEdge
python run.py --dataset OpTC -t 5 -d 2 -i DynTEdge
python run.py --dataset OpTC -t 5 -d 2.5 -i DynTEdge
python run.py --dataset OpTC -t 5 -d 3 -i DynTEdge
python run.py --dataset OpTC -t 5 -d 4 -i DynTEdge
mv results/stats.txt results/OpTC_dyn_softmax_gru.txt

python run.py --dataset OpTC -t 5 -d 0.125 -i DynTEdge
python run.py --dataset OpTC -t 5 -d 0.25 -i DynTEdge
python run.py --dataset OpTC -t 5 -d 0.5 -i DynTEdge -r LSTM
python run.py --dataset OpTC -t 5 -d 1 -i DynTEdge -r LSTM
python run.py --dataset OpTC -t 5 -d 1.5 -i DynTEdge -r LSTM
python run.py --dataset OpTC -t 5 -d 2 -i DynTEdge -r LSTM
python run.py --dataset OpTC -t 5 -d 2.5 -i DynTEdge -r LSTM
python run.py --dataset OpTC -t 5 -d 3 -i DynTEdge -r LSTM
python run.py --dataset OpTC -t 5 -d 4 -i DynTEdge -r LSTM
mv results/stats.txt results/OpTC_dyn_softmax_lstm.txt

python run.py -t 5 -d 0.5 -i DynTEdge
python run.py -t 5 -d 1 -i DynTEdge
python run.py -t 5 -d 1.5 -i DynTEdge
python run.py -t 5 -d 2 -i DynTEdge
python run.py -t 5 -d 2.5 -i DynTEdge
python run.py -t 5 -d 3 -i DynTEdge
python run.py -t 5 -d 4 -i DynTEdge
mv results/stats.txt results/LANL_dyn_softmax_gru.txt

python run.py -t 5 -d 0.5 -i DynTEdge -r LSTM
python run.py -t 5 -d 1 -i DynTEdge -r LSTM
python run.py -t 5 -d 1.5 -i DynTEdge -r LSTM
python run.py -t 5 -d 2 -i DynTEdge -r LSTM
python run.py -t 5 -d 2.5 -i DynTEdge -r LSTM
python run.py -t 5 -d 3 -i DynTEdge -r LSTM
python run.py -t 5 -d 4 -i DynTEdge -r LSTM
mv results/stats.txt results/LANL_dyn_softmax_lstm.txt