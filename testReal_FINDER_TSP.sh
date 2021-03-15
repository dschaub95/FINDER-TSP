#    prompt="Step3:testing the model of "$modelName" by using the synthetic datasets in the paper"
#    echo -e "\033[40;37m ${prompt} \033[0m" 
#    python -u "./"$modelName"/testSynthetic.py"

prompt="Testing the model by using the real datasets in the paper"
echo -e "\033[40;37m ${prompt} \033[0m" 
python -u testReal.py

prompt="Program is finished！"
echo -e "\033[40;37m ${prompt} \033[0m" 