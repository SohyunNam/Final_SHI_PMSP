import json
import pandas as pd

from agent.ppo import *
from environment.test_env import *
from cfg import *

torch.manual_seed(42)
random.seed(42)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == "__main__":
    cfg = get_cfg()

    if cfg.env == "OE":
        from environment.env import *
    elif cfg.env == "EE":
        from environment.env_2 import *

    num_block_list = [80, 160, 240]
    ddt_list = [0.8, 1.0, 1.2]
    pt_var_list = [0.1, 0.2, 0.3, 0.4, 0.5]

    weight = {80: {"ATCS": [5.88, 0.95], "COVERT": 6.605},
              160: {"ATCS": [6.605, 0.874], "COVERT": 1.4},
              240: {"ATCS": [7.053, 0.848], "COVERT": 0.9}}

    trained_model = ["SSPT", "ATCS", "MDD", "COVERT"]
    simulation_dir = './output/test_ppo_ep1/simulation' if not cfg.use_vessl else "/simulation"
    if not os.path.exists(simulation_dir):
        os.makedirs(simulation_dir)

    for num_block in num_block_list:
        if cfg.use_vessl:
            import vessl
            vessl.init(organization="sun-eng-dgx", project="Final-SHI-PMSP", hp=cfg)

        with open("SHI_sample.json", 'r') as f:
            block_sample = json.load(f)
        with open("SHI_test_sample_{0}_new.json".format(num_block), 'r') as f:
            sample_data = json.load(f)

        for ddt in ddt_list:
            for pt_var in pt_var_list:
                print("num_block = {0}, pt_var = {1}, ddt = {2}".format(num_block, pt_var, ddt))
                output_dict = dict()

                for test_i in sample_data.keys():
                    output_dict[test_i] = dict()
                    test_data = sample_data[test_i]
                    for model in trained_model:
                        print("    {0} | Model = {1}".format(test_i, model))
                        simulation_dir_rule = simulation_dir + '/{0}_{1}_{2}'.format(num_block, str(round(pt_var, 1)), str(round(ddt, 1)))
                        if not os.path.exists(simulation_dir_rule):
                            os.makedirs(simulation_dir_rule)

                        if model not in ["SSPT", "ATCS", "MDD", "COVERT"]:  # 강화학습으로 학습한 모델
                            tard_list = list()
                            setup_list = list()

                            for i in range(20):
                                env = WeldingLine(num_block=num_block,
                                                  block_sample=block_sample,
                                                  rule_weight=weight[num_block],
                                                  log_dir=simulation_dir_rule + '/rl_{0}_episode_{1}_{2}.csv'.format(
                                                      model, test_i, i),
                                                  test_sample=test_data,
                                                  pt_var=pt_var,
                                                  ddt=ddt,
                                                  is_train=False)

                                model_path = "trained_model/{0}.pt".format(model)
                                agent = PPO(cfg, env.state_size, env.action_size).to(device)
                                checkpoint = torch.load(model_path)
                                agent.load_state_dict(checkpoint["model_state_dict"])

                                state = env.reset()
                                done = False

                                while not done:
                                    logit = agent.pi(torch.from_numpy(state).float().to(device))
                                    prob = torch.softmax(logit, dim=-1)

                                    action = torch.argmax(prob).item()
                                    next_state, reward, done = env.step(action)
                                    state = next_state

                                    if done:
                                        log = env.get_logs(
                                            simulation_dir_rule + '/rl_{0}_episode_{1}_{2}.csv'.format(model, test_i,
                                                                                                       i))
                                        tard_list.append(sum(env.monitor.tardiness) / env.model["Sink"].total_finish)
                                        setup_list.append(env.monitor.setup / env.model["Sink"].total_finish)
                                        break

                            output_dict[test_i][model] = {"Tardiness": np.mean(tard_list),
                                                          "Setup": np.mean(setup_list)}

                        else:  # Heuristic rule
                            tard, setup = test(num_block=num_block, block_sample=block_sample,
                                               sample_data=sample_data, routing_rule=model,
                                               file_path=simulation_dir_rule + '/{0}_episode_'.format(model),
                                               ddt=ddt, pt_var=pt_var)
                            output_dict[test_i][model] = {"Tardiness": tard, "Setup": setup}

                with open(simulation_dir + "/Test Output({0}, {1}, {2}).json".format(num_block, str(round(pt_var, 1)),
                                                                                     str(round(ddt, 1))), 'r') as f:
                    json.dump(output_dict, f)

                temp = {model: {"Tardiness": 0.0, "Setup": 0.0} for model in trained_model}
                num_test = len([test_i for test_i in output_dict.keys()])
                for test_i in output_dict.keys():
                    for model in output_dict[test_i].keys():
                        temp[model]["Tardiness"] += output_dict[test_i][model]["Tardiness"] / num_test
                        temp[model]["Setup"] += output_dict[test_i][model]["Setup"] / num_test

                temp_df = pd.DataFrame(temp)
                temp_df.transpose()
                temp_df.to_excel(simulation_dir + "/Test_{0}_{1}_{2}.xlsx".format(num_block, str(round(pt_var, 1)),
                                                                                  str(round(ddt, 1))))