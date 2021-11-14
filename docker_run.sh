nvidia-docker run -it --volume $PWD:/home/FINDER-TSP --name finder-container finder:tsp bash
# nvidia-docker run -it --rm --volume $HOME/Desktop/TSP_code/graph_comb_opt:/home/S2V-DQN --name s2v-dqn-container s2v-dqn:test bash
# nvidia-docker run -it -v data:/home/S2V-DQN/data --name s2v-dqn-container s2v-dqn:test bash