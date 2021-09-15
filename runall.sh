python run.py -t 5 -d 0.25
python run.py -t 5 -d 0.5
python run.py -t 5 -d 1
python run.py -t 5 -d 2
python run.py -t 5 -d 3
python run.py -t 5 -d 4
mv results/stats.txt results/LANL_static.txt
python run.py -t 5 -d 1 -i TEdge
python run.py -t 5 -d 2 -i TEdge
python run.py -t 5 -d 3 -i TEdge
python run.py -t 5 -d 4 -i TEdge
mv results/stats.txt results/LANL_Softmax_v2.txt