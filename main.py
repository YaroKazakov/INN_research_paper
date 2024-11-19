from gymnasium.spaces import Discrete
from src.TRPO.optimize import *
import src.INN.get_INN_model as get_INN
import os
import argparse
from PostProcesses import run_scan
import pathlib
from sb.get_TRPO_model import create_dataset
N_expert = 100

def prepare_data():
    """
    collect dataset for training and validation
    """
    model_path = f"data/{Env_name}/{Env_name}_{N_expert}_TRPO.pth"
    pathlib.Path(f"data/{Env_name}").mkdir(exist_ok=True)
    # get dataset for following INN training:
    os.system(f"rm data/{Env_name}/data_training.csv")
    os.system(f"rm data/{Env_name}/data_training_val.csv")
    test(model_name=model_path, nrollout=150,
         file_ = f"data/{Env_name}/data_training.csv",
         sample_=False, agent=make_agent(N_expert))
        # get dataset for following INN validation:
    test(model_name=model_path, nrollout=15,
         file_= f"data/{Env_name}/data_training_val.csv",
         sample_=False, agent=make_agent(N_expert))

def train_student(n_neurons=40, isINN=True):
    name_ = "NN"
    if isINN: name_ = "INN"
    # train INN model:
    get_INN.train(make_agent(n_neurons), epochs = 20, batch_size = 128, \
          csv_to_train = f'data/{Env_name}/data_training.csv', \
          csv_to_val = f"data/{Env_name}/data_training_val.csv", \
          log_dir = f"data/{Env_name}/logs/Lunar_distill_{n_neurons}_" + name_, \
          path_to_model_for_save = f"data/{Env_name}/{Env_name}_" + name_ + f"_{n_neurons}.pth",
          isINN=isINN)


def run_inn_lunar_lander():
    test(model_name=f"data/{Env_name}/test_model_100.pth", nrollout=15000,
         file_="", sample_=False, render=True, agent=make_agent(n_neurons = 40))

def train_models_TRPO(nepochs = 100):
    '''
    run TRPO training
    '''
    pathlib.Path(f"data/{Env_name}").mkdir(exist_ok=True)
    env = gym.make(Env_name, render_mode="rgb_array")
    model_path = f"data/{Env_name}/{Env_name}_{N_expert}_TRPO.pth"
    #TRPO training
    agent = make_agent(N_expert)
    train(epochs=nepochs, model_name=model_path, metrics=f"data/{Env_name}/metric_TRPO_{N_expert}.csv", agent=agent)


def run_test():
    train_models_TRPO(nepochs=100)
    for k in range(10):
        #create_dataset(file_name="data_training.csv", render_mode="rgb_array", Nepisodes=50)
        #create_dataset(file_name="data_training_val.csv", render_mode="rgb_array", Nepisodes=1)
        prepare_data()
        train_student(n_neurons=100,isINN=True)
        train_student(n_neurons=100,isINN=False)
        run_scan(test_type="INN")
        run_scan(test_type="NN")
        os.system(f"scp data/{Env_name}/NN_reward.dat data/{Env_name}/NN_reward.dat_{k}")
        os.system(f"scp data/{Env_name}/INN_reward.dat data/{Env_name}/INN_reward.dat_{k}")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_INN_test", type= int, default=0, help="run INN test")
    parser.add_argument("--train_TRPO_expert", type= int, default=0, help="train TRPO expert")
    parser.add_argument("--train_INN_agent", type= int, default=0, help="train INN agent")
    args = parser.parse_args()
    if args.run_INN_test:
        run_inn_lunar_lander()
    if args.train_TRPO_expert:
        train_models_TRPO()
    if args.train_INN_agent:
        train_student(N_expert, isINN=True)
    #run_test()
    #train_student(100, isINN=True)
    #train_models_TRPO()
    #train_models_TRPO()
    #prepare_data()
    #for i in [100]:
    #    train_student(i,True)
    #for i in [90,80,70,60,50,40,30,20,15,10]:
    #    train_student(i,False)
    #run_inn_lunar_lander()
    #run_test()
    #prepare_data()

    #train_student(N_expert, isINN=True)
    #run_scan(test_type="INN")
    #train_student(N_expert, isINN=False)
    #run_scan(test_type="NN")
