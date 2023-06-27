# import vessl
from cfg import *
from agent.ppo import *  # 학습 알고리즘
  # 학습 환경

torch.manual_seed(42)
random.seed(42)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


if __name__ == "__main__":
    cfg = get_cfg()
    if cfg.use_vessl:
        vessl.init(organization="snu-eng-gtx1080", project="pmsp", hp=cfg)
        if cfg.env == "OE":
            from environment.env import WeldingLine
        elif cfg.env == "EE":
            from environment.env_2 import WeldingLine
    else:
        from environment.env import WeldingLine

    load_model = False
    num_block = 80
    weight_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.0]
    weight = {80: {"ATCS": [5.88, 0.95], "COVERT": 6.605},
              160: {"ATCS": [6.605, 0.874], "COVERT": 1.4},
              240: {"ATCS": [7.053, 0.848], "COVERT": 0.9}}

    for weight in weight_list:
        weight_tard = weight
        weight_setup = 1 - weight
        lr = 0.0001
        optim = "Adam" if not cfg.use_vessl else cfg.optim
        eps_clip = 0.2
        num_episode = 50000

        with open('SHI_sample.json', 'r') as f:
            block_sample = json.load(f)

        model_dir = './output/train_ddt_{0}_{1}_July/model/'.format(int(10 * weight), math.ceil(
            10 * (1 - weight))) if not cfg.use_vessl else "/output/{0}_{1}/model".format(int(10 * weight),
                                                                                         math.ceil(10 * (1 - weight)))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        simulation_dir = './output/train_ddt_{0}_{1}_July/simulation/'.format(int(10 * weight), math.ceil(
            10 * (1 - weight))) if not cfg.use_vessl else "/output/{0}_{1}/simulation".format(int(10 * weight),
                                                                                              math.ceil(
                                                                                                  10 * (1 - weight)))
        if not os.path.exists(simulation_dir):
            os.makedirs(simulation_dir)

        log_dir = './output/train_ddt_{0}_{1}_July/log/'.format(int(10 * weight), math.ceil(
            10 * (1 - weight))) if not cfg.use_vessl else "/output/{0}_{1}/log".format(int(10 * weight),
                                                                                              math.ceil(
                                                                                                  10 * (1 - weight)))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        env = WeldingLine(num_block=num_block, reward_weight=[weight_tard, weight_setup], block_sample=block_sample,
                          rule_weight=weight[num_block], is_train=True)

        agent = PPO(cfg, env.state_size, env.action_size, lr=lr, eps_clip=eps_clip, optimizer_name=optim).to(device)

        if cfg.load_model:
            checkpoint = torch.load('./output/train_ddt/model/episode-10001.pt')
            start_episode = checkpoint['episode'] + 1
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            start_episode = 1

        with open(log_dir + "train_log.csv", 'w') as f:
            f.write('episode, reward, reward_tard, reward_setup, tardiness, setup, makespan\n')
        # with open(log_dir + "validation_log.csv", 'w') as f:
        #     f.write('episode, tardiness, setup_ratio, makespan\n')

        test_data = 'SHI_test_sample_{0}_new.json'.format(num_block)
        with open(test_data, 'r') as f:
            test_sample_data = json.load(f)

        for episode in range(start_episode, start_episode + num_episode + 1):
            state = env.reset()
            r_epi = 0.0
            done = False

            while not done:
                for t in range(cfg.T_horizon):
                    logit = agent.pi(torch.from_numpy(state).float().to(device))
                    prob = torch.softmax(logit, dim=-1)

                    m = Categorical(prob)
                    action = m.sample().item()
                    next_state, reward, done = env.step(action)

                    agent.put_data((state, action, reward, next_state, prob[action].item(), done))
                    state = next_state

                    r_epi += reward
                    if done:
                        # print("episode: %d | reward: %.4f" % (e, r_epi))
                        # print("episode: %d | total_rewards: %.2f" % (episode, r_epi))
                        tardiness = np.mean(env.monitor.tardiness)
                        setup = env.monitor.setup / env.model["Sink"].total_finish
                        # makespan = max(test_env.monitor.throughput.keys())
                        print("{0}_{1} |".format(int(10 * weight), int(10 * (1 - weight))),
                              "episode: %d | reward: %.4f | Setup: %.4f | Tardiness %.4f" % (
                                  episode, r_epi, setup, tardiness))
                        with open(log_dir + "train_log.csv", 'a') as f:
                            f.write('%d,%.2f,%.2f,%.2f, %.4f, %.4f \n' % (episode, r_epi, env.tard_reward, env.setup_reward, tardiness, setup))
                        break
                agent.train_net()

            if episode % 50 == 0 or episode == 1:
                tardiness = np.mean(env.monitor.tardiness)
                setup = env.monitor.setup / env.model["Sink"].total_finish
                # makespan = max(test_env.monitor.throughput.keys())
                if cfg.use_vessl:
                    vessl.log(step=episode, payload={'reward': r_epi})
                    vessl.log(step=episode, payload={'reward_setup': env.setup_reward})
                    vessl.log(step=episode, payload={'reward_tard': env.tard_reward})
                    vessl.log(step=episode, payload={'Train_Tardiness': tardiness})
                    vessl.log(step=episode, payload={'Train_Setup': setup})

                # _ = env.get_logs(simulation_dir + "/log_{0}.csv".format(episode))
                agent.save(episode, model_dir)

            # if episode % 100 == 1:
            #     test_tardiness, test_setup_ratio, test_makespan = evaluate(episode, agent, simulation_dir, test_sample_data)
            #     with open(log_dir + "validation_log.csv", 'a') as f:
            #         f.write('%d,%.4f,%.4f,%.4f \n' % (episode, test_tardiness, test_setup_ratio, test_makespan))
                #
                # vessl.log(step=episode, payload={'Test_Tardiness': test_tardiness})
                # vessl.log(step=episode, payload={'Test_Setup': test_setup_ratio})
                # vessl.log(step=episode, payload={'Test_Makespan': test_makespan})

