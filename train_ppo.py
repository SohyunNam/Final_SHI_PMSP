# import vessl
from cfg import *
from agent.ppo import *  # 학습 알고리즘
from environment.env import WeldingLine  # 학습 환경

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def evaluate(episode, agent, simulation_dir, test_sample_data):
    tardiness_list = list()
    setup_list = list()
    makespan_list = list()
    with torch.no_grad():
        for test_key in test_sample_data.keys():
            test_env = WeldingLine(
                num_block=num_block,
                block_sample=block_sample,
                log_dir=simulation_dir + '/episode_{0}_{1}_'.format(episode, test_key),
                test_sample=test_sample_data[test_key])

            done = False
            test_state = test_env.reset()
            while not done:
                logit = agent.pi(torch.from_numpy(test_state).float().to(device))
                prob = torch.softmax(logit, dim=-1)

                test_action = torch.argmax(prob).item()
                test_next_state, reward, done = test_env.step(test_action)

                # model.put_data((test_state, test_action, reward, test_next_state, prob[test_action].item(), done))
                test_state = test_next_state

                if done:
                    _ = test_env.monitor.get_logs(simulation_dir + '/episode_{0}_{1}.csv'.format(episode, test_key))
                    break

            tardiness_list.append(sum(test_env.monitor.tardiness) / test_env.num_block)
            setup_list.append(test_env.monitor.setup / test_env.model["Sink"].total_finish)
            makespan_list.append(test_env.model["Sink"].makespan)

    tardiness = np.mean(tardiness_list)
    setup_ratio = np.mean(setup_list)
    makespan = np.mean(makespan_list)

    return tardiness, setup_ratio, makespan


if __name__ == "__main__":
    cfg = get_cfg()
    # vessl.init(organization="snu-eng-gtx1080", project="pmsp", hp=cfg)

    load_model = False
    num_block = 80
    weight_list = [0.5, 0.6, 0.7]
    for weight in weight_list:
        weight_tard = weight
        weight_setup = 1 - weight
        learning_rate = 0.0005
        gamma = cfg.gamma
        lmbda = cfg.lmbda
        eps_clip = cfg.eps_clip
        K_epoch = cfg.K_epoch
        T_horizon = cfg.T_horizon

        num_episode = 10000

        score_avg = 0

        with open('SHI_sample.json', 'r') as f:
            block_sample = json.load(f)

        model_dir = './output/train_ddt_{0}_{1}_new_ppo/model/'.format(int(10 * weight), int(10 * (1 - weight)))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        simulation_dir = './output/train_ddt_{0}_{1}_new_ppo/simulation/'.format(int(10 * weight), int(10 * (1 - weight)))
        if not os.path.exists(simulation_dir):
            os.makedirs(simulation_dir)

        log_dir = './output/train_ddt_{0}_{1}_new_ppo/log/'.format(int(10 * weight), int(10 * (1 - weight)))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        env = WeldingLine(num_block=num_block, reward_weight=[weight_tard, weight_setup], block_sample=block_sample)

        agent = PPO(env.state_size, env.action_size, learning_rate, gamma, lmbda, eps_clip, K_epoch, T_horizon).to(device)

        # if cfg.load_model:
        # checkpoint = torch.load('./output/train_ddt/model/episode-10001.pt')
        # start_episode = checkpoint['episode'] + 1
        # agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # # else:
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
                for t in range(T_horizon):
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
                        tardiness = (np.sum(env.monitor.tardiness) / env.num_block) * 24
                        setup = env.monitor.setup / env.model["Sink"].total_finish
                        # makespan = max(test_env.monitor.throughput.keys())
                        makespan = env.model["Sink"].makespan
                        print("episode: %d | reward: %.4f | Setup: %.4f | Tardiness %.4f | makespan %.4f" % (
                            episode, r_epi, setup, tardiness, makespan))
                        with open(log_dir + "train_log.csv", 'a') as f:
                            f.write('%d,%.2f,%.2f,%.2f, %.4f, %.4f,%.4f \n' % (episode, r_epi, env.tard_reward, env.setup_reward, tardiness, setup, makespan))
                        break
                agent.train_net()

            if episode % 50 == 0 or episode == 1:
                tardiness = (np.sum(env.monitor.tardiness) / env.num_block) * 24
                setup = np.sum(env.monitor.setup_list) / env.model["Sink"].total_finish
                # makespan = max(test_env.monitor.throughput.keys())
                makespan = env.model["Sink"].makespan

                # vessl.log(step=episode, payload={'reward': r_epi})
                # vessl.log(step=episode, payload={'reward_setup': env.setup_reward})
                # vessl.log(step=episode, payload={'reward_tard': env.tard_reward})
                # vessl.log(step=episode, payload={'Train_Tardiness': tardiness})
                # vessl.log(step=episode, payload={'Train_Setup': setup})
                # vessl.log(step=episode, payload={'Train_Makespan': makespan})

                _ = env.get_logs(simulation_dir + "/log_{0}.csv".format(episode))
                agent.save(episode, model_dir)

            # if episode % 100 == 1:
            #     test_tardiness, test_setup_ratio, test_makespan = evaluate(episode, agent, simulation_dir, test_sample_data)
            #     with open(log_dir + "validation_log.csv", 'a') as f:
            #         f.write('%d,%.4f,%.4f,%.4f \n' % (episode, test_tardiness, test_setup_ratio, test_makespan))
                #
                # vessl.log(step=episode, payload={'Test_Tardiness': test_tardiness})
                # vessl.log(step=episode, payload={'Test_Setup': test_setup_ratio})
                # vessl.log(step=episode, payload={'Test_Makespan': test_makespan})

