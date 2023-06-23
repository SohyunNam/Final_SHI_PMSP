import json
import simpy
import vessl
import copy
import numpy as np
from environment.simulation import *

np.random.seed(42)
random.seed(42)

class WeldingLine:
    def __init__(self, num_block=80, num_line=3, block_sample=None,
                 start_lag=0, log_dir=None, reward_weight=None, test_sample=None,
                 ddt=1.0, pt_var=0.1, is_train=True):
        self.num_block = num_block
        self.num_line = num_line
        self.start_lag = start_lag
        self.log_dir = log_dir
        self.reward_weight = reward_weight if reward_weight is not None else [0.5, 0.5]
        self.test_sample = test_sample
        self.block_sample = block_sample
        self.ddt = ddt
        self.pt_var = pt_var
        self.is_train = is_train

        self.state_size = 14
        self.action_size = 4

        self.tard_reward = 0.0
        self.setup_reward = 0.0
        self.msp_reward = 0.0

        self.done = False
        self.e = 0
        self.time = 0
        self.num_jobs = 0
        self.sim_block = dict()

        self.time_list = list()
        self.tardiness_ratio_list = list()
        self.setup_ratio_list = list()

        self.sim_env, self.model, self.routing, self.monitor = self._modeling()

    def step(self, action):
        done = False
        self.previous_time_step = self.sim_env.now

        self.routing.decision.succeed(action)
        self.routing.indicator = False

        while True:
            if self.routing.indicator:
                if self.sim_env.now != self.time:
                    self.time = self.sim_env.now
                break

            if self.num_jobs == self.model["Sink"].total_finish:
                done = True
                self.sim_env.run()
                # if self.e % 50 == 0:
                #     self.monitor.save_tracer()
                # # self.monitor.save_tracer()
                break
            if len(self.sim_env._queue) == 0:
                self.monitor.get_logs(file_path="Test_ddt.csv")
                print("Break!")
            self.sim_env.step()

        reward = self._calculate_reward()
        next_state = self._get_state()

        return next_state, reward, done

    def reset(self):
        self.sim_block = dict()
        self.done = False
        self.tard_reward = 0.0
        self.setup_reward = 0.0
        if self.is_train:
            self.pt_var = np.random.uniform(low=0.1, high=0.5)
            self.ddt = np.random.uniform(low=0.8, high=1.2)
        self.sim_env, self.model, self.routing, self.monitor = self._modeling()

        self.monitor.reset()
        for name in self.model.keys():
            if name != "Source":
                self.model[name].reset()
        self.routing.reset()

        while True:
            # Check whether there is any decision time step
            if self.routing.indicator:
                break
            self.sim_env.step()

        return self._get_state()

    def _modeling(self):
        # data modeling
        if self.test_sample is None:
            if self.num_block == 240:
                week3_due_date = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19]
                due_date_list = np.random.choice(week3_due_date, size=self.num_block)
                iat = (960 * 18) / self.num_block
            elif self.num_block == 160:
                week2_due_date = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
                due_date_list = np.random.choice(week2_due_date, size=self.num_block)
                iat = (960 * 12) / self.num_block
            else:
                due_date_list = list(np.random.randint(low=0, high=6, size=self.num_block))  # Week 1 버전
                iat = (960 * 6) / self.num_block
        else:
            due_date_list = self.test_sample["due_date"]
            if self.num_block == 240:
                iat = (960 * 18) / self.num_block
            elif self.num_block == 160:
                iat = (960 * 12) / self.num_block
            else:
                iat = (960 * 6) / self.num_block

        # Block Sampling
        if self.test_sample is None:
            block_list = np.random.choice([key for key in self.block_sample.keys()], size=self.num_block)
        else:
            block_list = self.test_sample['block_list']

        # simulation object modeling
        model = dict()
        env = simpy.Environment()
        monitor = Monitor()
        monitor.reset()

        self.sim_block = dict()
        self.num_jobs = 0
        create_dict = dict()
        # Steel class로 모델링 + self.sim_block에 블록 저장 + self.num_jobs 계산
        for block_idx in range(len(block_list)):
            block_name = block_list[block_idx]
            # block_due_date = due_date_list[block_idx]
            block_data = self.block_sample[block_name]
            total_pt = 0.0
            for steel_name in block_data.keys():
                avg_speed = 1200 - ((block_data[steel_name]["weld_size"] - 4.5) / 0.5) * 50
                total_pt = (block_data[steel_name]["length"] / avg_speed) * block_data[steel_name]["num_steel"]

            create_time = due_date_list[block_idx]
            block_due_date = math.ceil(create_time + (total_pt * self.ddt) / (24 * 60))
            self.sim_block["Block_{0}".format(block_idx)] = dict()
            self.sim_block["Block_{0}".format(block_idx)]["due_date"] = block_due_date
            self.sim_block["Block_{0}".format(block_idx)]["num_steel"] = 0

            steel_idx = 0
            block_steel_list = list()
            for steel_name in self.block_sample[block_name].keys():
                for i in range(self.block_sample[block_name][steel_name]["num_steel"]):
                    self.sim_block["Block_{0}".format(block_idx)][
                        "Steel_{0}_{1}_{2}".format(block_idx, steel_idx, i)] = Steel(
                        name="Steel_{0}_{1}_{2}".format(block_idx, steel_idx, i),
                        block="Block_{0}".format(block_idx), steel="Steel_{0}_{1}".format(block_idx, steel_idx),
                        feature=self.block_sample[block_name][steel_name], due_date=block_due_date)

                    block_steel_list.append(copy.deepcopy(self.sim_block["Block_{0}".format(block_idx)][
                                                              "Steel_{0}_{1}_{2}".format(block_idx, steel_idx, i)]))
                    self.sim_block["Block_{0}".format(block_idx)]["num_steel"] += 1
                    self.num_jobs += 1
                steel_idx += 1
            if create_time not in create_dict.keys():
                create_dict[create_time] = list()
            create_dict[create_time].append(copy.deepcopy(block_steel_list))

        routing = Routing(env, model, self.sim_block, monitor, self.num_jobs)
        model["Source"] = Source(env, create_dict, routing, iat, monitor)

        for i in range(self.num_line):
            model["Line {0}".format(i)] = Process(env, "Line {0}".format(i), model, routing, monitor, pt_var=self.pt_var)
            model["Line {0}".format(i)].reset()
        model["Sink"] = Sink(env, self.sim_block, monitor)
        model["Sink"].reset()

        return env, model, routing, monitor

    def _get_state(self):
        # define 4 features
        f_1 = np.zeros(self.num_line)  # Setup -> 현재 라인의 셋업값과 같은 셋업인 부재의 수
        f_2 = np.zeros(4)  # Due Date -> Tardiness Level for non-setup
        f_3 = np.zeros(4)  # Due Date -> Tardiness Level for setup
        f_4 = np.zeros(self.num_line)  # General Info -> 각 라인에서의 남은 작업시간

        input_queue = copy.deepcopy(self.routing.queue.items)

        # Feature 1, 4
        for line_num in range(self.num_line):
            line = self.model["Line {0}".format(line_num)]
            line_feature = line.setup
            same_setup_list = [1 for job in input_queue if job.web_face == line_feature]
            f_1[line_num] = np.sum(same_setup_list) / len(input_queue) if len(input_queue) > 0 else 0.0

            if line.job is not None and not line.idle:
                f_4[line_num] = (line.planned_finish_time - self.sim_env.now) / (line.planned_finish_time - line.start_time)

        # Feature 2, 3
        calling_line = self.model[self.routing.line]
        setting = calling_line.setup

        non_setup_list = list()
        setup_list = list()

        if len(input_queue) > 0:
            for job in input_queue:
                if job.web_face == setting:
                    non_setup_list.append(job)
                else:
                    setup_list.append(job)

        def _cal_expected_finish_time(var, job_list):
            expected_time = self.sim_env.now
            for job in job_list:
                pt = job.avg_pt * var
                if (expected_time + pt) % 1440 <= 960:
                    expected_time += pt
                else:
                    day = math.floor(expected_time / 1440)
                    next_day = day + 1 if day % 7 != 5 else day + 2
                    # next_day = day + 1
                    expected_time = next_day * 1440 + pt

            return expected_time

        # Feature 2
        if len(non_setup_list) > 0:
            g_1 = 0
            g_2 = 0
            g_3 = 0
            g_4 = 0

            for non_setup_job in non_setup_list:
                job_dd = non_setup_job.due_date * 1440 + 960
                # finished_jobs = self.model["Sink"].finished[non_setup_job.block]["num"] if non_setup_job.block in \
                #                                                                     self.model[
                #                                                                         "Sink"].finished.keys() else 0

                # num_residual = self.sim_block[non_setup_job.block]["num_steel"] - finished_jobs
                job_list = [job for job in input_queue if job.block == non_setup_job.block]
                max_tightness = job_dd - _cal_expected_finish_time(1 + self.pt_var, job_list)
                min_tightness = job_dd - _cal_expected_finish_time(1 - self.pt_var, job_list)

                if max_tightness > 0:
                    g_1 += 1
                elif (max_tightness <= 0) and (min_tightness > 0):
                    g_2 += 1
                elif (min_tightness <= 0) and (self.sim_env.now > job_dd):
                    g_3 += 1
                elif self.sim_env.now < job_dd:
                    g_4 += 1
                else:
                    print(0)

            f_2[0] = g_1 / len(non_setup_list)
            f_2[1] = g_2 / len(non_setup_list)
            f_2[2] = g_3 / len(non_setup_list)
            f_2[3] = g_4 / len(non_setup_list)

            # Feature 3
            if len(setup_list) > 0:
                g_1 = 0
                g_2 = 0
                g_3 = 0
                g_4 = 0

                for setup_job in setup_list:
                    job_dd = setup_job.due_date * 1440 + 960

                    job_list = [job for job in input_queue if job.block == setup_job.block]
                    max_tightness = job_dd - _cal_expected_finish_time(1 + self.pt_var, job_list)
                    min_tightness = job_dd - _cal_expected_finish_time(1 - self.pt_var, job_list)

                    if max_tightness > 0:
                        g_1 += 1
                    elif (max_tightness <= 0) and (min_tightness > 0):
                        g_2 += 1
                    elif (min_tightness <= 0) and (self.sim_env.now > job_dd):
                        g_3 += 1
                    elif self.sim_env.now < job_dd:
                        g_4 += 1
                    else:
                        print(0)

                f_3[0] = g_1 / len(setup_list)
                f_3[1] = g_2 / len(setup_list)
                f_3[2] = g_3 / len(setup_list)
                f_3[3] = g_4 / len(setup_list)

        state = np.concatenate((f_1, f_2, f_3, f_4), axis=None)
        return state

    def _calculate_reward(self):
        self.reward = 0
        # setup
        if self.routing.setup:
            self.reward -= 0.1 * self.reward_weight[1]
            self.setup_reward -= -0.1 * self.reward_weight[1]

        # Earliness / Tardiness
        for difference in self.model["Sink"].finished_block:
            self.reward += (np.exp(- difference * 24 * 60) - 1) * self.reward_weight[0]
            self.tard_reward += (np.exp(- difference * 24 * 60) - 1) * self.reward_weight[0]
            # elif difference_time > 0:  # earliness, vessl experiment 5, 6, 7, 10에는 존재
            #     self.reward += np.exp(-difference_time) - 1
        self.model["Sink"].finished_block = list()

        # # makespan
        # self.msp_reward = (4 + int((self.num_block / 80))) * (self.routing.created / self.num_jobs) - self.model["Sink"].makespan0

        return self.reward

    def get_logs(self, path=None):
        log = self.monitor.get_logs(path)
        return log
