from gymnasium.spaces import Discrete
from src.TRPO.optimize import *
import src.INN.get_INN_model as get_INN
import os
import argparse
from PostProcesses import run_scan, plot_reward
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

def train_student(n_neurons=(40,40), isINN=True, isKAN = False):
    name_ = "NN"
    if isINN: name_ = "INN"
    if isKAN:
        name_ = "KAN"
        isINN=False

    # train INN model:
    get_INN.train(make_agent(n_neurons, isKAN=isKAN), epochs = 20, batch_size = 128, \
          csv_to_train = f'data/{Env_name}/data_training.csv', \
          csv_to_val = f"data/{Env_name}/data_training_val.csv", \
          log_dir = f"data/{Env_name}/logs/Lunar_distill_{n_neurons}_" + name_, \
          path_to_model_for_save = f"data/{Env_name}/{Env_name}_" + name_ + f"_{n_neurons[0]}.pth",
          isINN=isINN, isKAN=isKAN)


def run_inn_lunar_lander():
    test(model_name=f"data/{Env_name}/{Env_name}_INN_100.pth100_0_0", nrollout=1,
         file_="", sample_=False, render=True, agent=make_agent(n_neurons = (100,100), isKAN=False))

def train_models_TRPO(nepochs = 100):
    '''
    run TRPO training
    '''
    pathlib.Path(f"data/{Env_name}").mkdir(exist_ok=True)
    model_path = f"data/{Env_name}/{Env_name}_{N_expert}_TRPO.pth"
    #TRPO training
    agent = make_agent(N_expert)
    train(epochs=nepochs, model_name=model_path, metrics=f"data/{Env_name}/metric_TRPO_{N_expert}.csv", agent=agent)


def run_test():
    #train_models_TRPO(nepochs=100)
    for k in range(0,5): # range(5,10):
        #try:
        #    create_dataset(file_name="data_training.csv", render_mode="rgb_array", Nepisodes=50)
        #    create_dataset(file_name="data_training_val.csv", render_mode="rgb_array", Nepisodes=1)
        #except:
        #    prepare_data()
        train_student(n_neurons=(100,100),isINN=True)
        run_scan(test_type="INN")
        train_student(n_neurons=(100,100),isINN=False)
        run_scan(test_type="NN")
        train_student(n_neurons=(100,100),isINN=False, isKAN=True)
        run_scan(test_type="KAN")
        os.system(f"scp data/{Env_name}/NN_reward.dat data/{Env_name}/NN_reward.dat_{k}")
        os.system(f"scp data/{Env_name}/INN_reward.dat data/{Env_name}/INN_reward.dat_{k}")
        os.system(f"scp data/{Env_name}/KAN_reward.dat data/{Env_name}/KAN_reward.dat_{k}")

        os.system(f"scp data/{Env_name}/NN_reward.datBICUBIC0 data/{Env_name}/NN_reward.datBICUBIC0_{k}")
        os.system(f"scp data/{Env_name}/NN_reward.datBICUBIC1 data/{Env_name}/NN_reward.datBICUBIC1_{k}")
        os.system(f"scp data/{Env_name}/NN_reward.datBICUBIC2 data/{Env_name}/NN_reward.datBICUBIC2_{k}")

        os.system(f"scp data/{Env_name}/INN_reward.datBICUBIC0 data/{Env_name}/INN_reward.datBICUBIC0_{k}")
        os.system(f"scp data/{Env_name}/INN_reward.datBICUBIC1 data/{Env_name}/INN_reward.datBICUBIC1_{k}")
        os.system(f"scp data/{Env_name}/INN_reward.datBICUBIC2 data/{Env_name}/INN_reward.datBICUBIC2_{k}")

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
