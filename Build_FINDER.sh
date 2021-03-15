set -e

modelName="FINDER_TSP"

prompt="Step1:Building the model of "$modelName""
echo -e "\033[40;37m ${prompt} \033[0m" 
python setup.py build_ext -i