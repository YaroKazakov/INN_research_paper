from src.TRPO.optimize import *
import src.INN.get_INN_model as get_INN

def whole_pipeline():
    model_path = "data/Lunar/LunarLander-v3_s.pth"
    #TRPO training
    agent = AgentTRPO(env.observation_space, n_actions, n_neurons=100)
    train(epochs=100, model_name=model_path, metrics="data/Lunar/metric_TRPO.csv")
    #get dataset for following INN training:
    test(model_name=model_path, nrollout=10,  file_ = "data/Lunar/data_training.csv", sample_=False)
    # get dataset for following INN validation:
    test(model_name=model_path, nrollout=10, file_="data/Lunar/data_training_val.csv", sample_=False)
    # train INN model
    get_INN.train(epochs = 10, batch_size = 128, \
          csv_to_train = 'data/Lunar/data_training.csv', \
          csv_to_val = "data/Lunar/data_training_val.csv", \
          log_dir = "data/Lunar/logs/Lunar_distill", \
          path_to_model_for_save = "data/Lunar/LunarLander-v3_INN.pth")
    test(model_name=model_path, nrollout=10, file_="", sample_=False)


def run_INN_Lunar_lander():
    test(model_name="data/test_mode0.pth", nrollout=10, file_="", sample_=False)

def train_models():
    env = gym.make(Env_name, render_mode="rgb_array")
    n_actions = 4
    for el in [2,5,15,20,30]:
        model_path = f"data/Lunar/LunarLander-v3_{el}.pth"
        #TRPO training
        agent = AgentTRPO(env.observation_space, n_actions, n_neurons=el)
        train(epochs=100, model_name=model_path, metrics=f"data/Lunar/metric_TRPO_{el}.csv", agent=agent)

if __name__=="__main__":
    #whole_pipeline()
    #run_INN_Lunar_lander()
    train_models()