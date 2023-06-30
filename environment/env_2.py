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
                 rule_weight=None, log_dir=None, reward_weight=None, test_sample=None,
                 ddt=1.0, pt_var=0.1, is_train=True):
        self.num_block = num_block
        self.num_line = num_line
        self.rule_weight = rule_weight
        self.log_dir = log_dir
        self.reward_weight = reward_weight if reward_weight is not None else [0.5, 0.5]
        self.test_sample = test_sample
        self.block_sample = block_sample
        self.ddt = ddt
        self.pt_var = pt_var
        self.is_train = is_train

        self.state_size = 14 + 4 * num_line
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
        iat = (960 * 6 * round(self.num_block / 80)) / self.num_block
        if self.test_sample is None:
            if self.num_block == 240:
                week3_due_date = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19]
                due_date_list = np.random.choice(week3_due_date, size=self.num_block)
                # iat = (960 * 18) / self.num_block
            elif self.num_block == 160:
                week2_due_date = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
                due_date_list = np.random.choice(week2_due_date, size=self.num_block)
                # iat = (960 * 12) / self.num_block
            else:
                due_date_list = list(np.random.randint(low=0, high=6, size=self.num_block))  # Week 1 버전
                # iat = (960 * 6) / self.num_block
        else:
            due_date_list = self.test_sample["due_date"]
            # if self.num_block == 240:
            #     iat = (960 * 18) / self.num_block
            # elif self.num_block == 160:
            #     iat = (960 * 12) / self.num_block
            # else:
            #     iat = (960 * 6) / self.num_block

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
                total_pt += (block_data[steel_name]["length"] / avg_speed) * block_data[steel_name]["num_steel"]

            create_time = due_date_list[block_idx]
            block_due_date = math.floor(create_time + ((total_pt * self.ddt) / (24 * 60)))
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

        routing = Routing(env, model, self.sim_block, monitor, self.num_jobs, self.rule_weight)
        model["Source"] = Source(env, create_dict, routing, iat, monitor)

        for i in range(self.num_line):
            model["Line {0}".format(i)] = Process(env, "Line {0}".format(i), model, routing, monitor, pt_var=self.pt_var)
            model["Line {0}".format(i)].reset()
        model["Sink"] = Sink(env, self.sim_block, monitor)
        model["Sink"].reset()

        return env, model, routing, monitor

    def _get_state(self):
        # define 9 features
        f_1 = np.zeros(4)  # tardiness level of jobs in rouitng.queue
        f_2 = np.zeros(3)  # tightness level min, avg, max
        f_3 = np.zeros(2)  # routing을 요청한 경우 해당 machine의 셋업값, queue에 있는 job 중 해당 셋업값과 같은 셋업값을 갖는 job의 수
        f_4 = np.zeros(self.num_line)  # 각 machine의 셋업 상태
        f_5 = np.zeros([self.num_line, 2])  # machine 별 각 셋업값을 가지는 job의 비율(routing.queue에 있는 job들 중에서) / 가능한 셋업 경우의 수 : 0~5, 6가지
        f_6 = np.zeros(1)  # completion rate
        f_7 = np.zeros(self.num_line)  # 각 machine에서의 progress rate
        f_8 = np.zeros(2)  # tardiness level index (x, v)
        f_9 = np.zeros(2)  # setup index (x, v)

        input_queue = copy.deepcopy(list(self.routing.queue.items))

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

        # Feature 1, 2
        tt_list = list()
        if len(input_queue) > 0:
            g_1 = 0
            g_2 = 0
            g_3 = 0
            g_4 = 0

            for job in input_queue:
                job_dd = job.due_date * 1440 + 960
                job_list = [q_job for q_job in input_queue if job.block == job.block]
                max_tightness = job_dd - _cal_expected_finish_time(1 + self.pt_var, job_list)
                min_tightness = job_dd - _cal_expected_finish_time(1 - self.pt_var, job_list)

                avg_tightness = job_dd - _cal_expected_finish_time(1, job_list)
                tt_list.append(avg_tightness)

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

            f_1[0] = g_1 / len(input_queue)
            f_1[1] = g_2 / len(input_queue)
            f_1[2] = g_3 / len(input_queue)
            f_1[3] = g_4 / len(input_queue)

        f_2[0] = np.min(tt_list) if len(tt_list) > 0 else 0.0
        f_2[1] = np.mean(tt_list) if len(tt_list) > 0 else 0.0
        f_2[2] = np.max(tt_list) if len(tt_list) > 0 else 0.0

        # f_3
        calling_line = self.model[self.routing.line]
        setting = calling_line.setup
        f_3[0] = setting / 200

        same_feature = 0
        for job in input_queue:
            if job.web_face == setting:
                same_feature += 1
        f_3[1] = same_feature / len(input_queue) if len(input_queue) > 0 else 0.0

        # f_4, 5, 7
        for line_num in range(self.num_line):
            line = self.model["Line {0}".format(line_num)]
            line_feature = line.setup
            f_4[line_num] = line_feature / 200

            same_setup_list = [1 for job in input_queue if job.web_face == line_feature]
            f_5[line_num][0] = np.sum(same_setup_list) / len(input_queue) if len(input_queue) > 0 else 0.0
            f_5[line_num][1] = 1 - f_5[line_num][0]

            if line.job is not None and not line.idle:
                f_7[line_num] = (self.sim_env.now - line.start_time) / (line.planned_finish_time - line.start_time)

        # f_6
        f_6[0] = self.model["Sink"].total_finish / self.num_jobs

        if self.sim_env.now > 0:
            self.time_list.append(self.sim_env.now)
            setup_ratio = self.monitor.setup / self.routing.created if self.routing.created > 0 else 0.0
            self.setup_ratio_list.append(setup_ratio)
            tardiness_ratio = sum(self.monitor.tardiness) / self.model["Sink"].total_finish if self.model["Sink"].total_finish > 0 else 0.0
            self.tardiness_ratio_list.append(tardiness_ratio)

        f_8[0] = self.setup_ratio_list[-1] if len(self.setup_ratio_list) else 0.0
        f_9[0] = self.tardiness_ratio_list[-1] if len(self.tardiness_ratio_list) else 0.0
        if len(self.time_list) > 1:
            v_setup = (self.setup_ratio_list[-1] - self.setup_ratio_list[-2]) / (
                        self.time_list[-1] - self.time_list[-2]) if self.time_list[-1] != self.time_list[-2] else 0.0
            f_8[1] = 1 / (1 + np.exp(-v_setup))
            v_tard = (self.tardiness_ratio_list[-1] - self.tardiness_ratio_list[-2]) / (
                    self.time_list[-1] - self.time_list[-2]) if self.time_list[-1] != self.time_list[-2] else 0.0
            f_9[1] = 1 / (1 + np.exp(-v_tard))

        state = np.concatenate((f_1, f_2, f_3, f_4, f_5, f_6, f_7, f_8, f_9), axis=None)
        return state

    def _calculate_reward(self):
        self.reward = 0
        # setup
        if self.routing.setup:
            self.reward -= 0.1 * self.reward_weight[1]
            self.setup_reward -= 0.1 * self.reward_weight[1]

        # Earliness / Tardiness
        for difference in self.model["Sink"].finished_block:
            self.reward += (np.exp(- difference) - 1) * self.reward_weight[0]
            self.tard_reward += (np.exp(- difference) - 1) * self.reward_weight[0]
            # elif difference_time > 0:  # earliness, vessl experiment 5, 6, 7, 10에는 존재
            #     self.reward += np.exp(-difference_time) - 1
        self.model["Sink"].finished_block = list()

        # # makespan
        # self.msp_reward = (4 + int((self.num_block / 80))) * (self.routing.created / self.num_jobs) - self.model["Sink"].makespan0

        return self.reward

    def get_logs(self, path=None):
        log = self.monitor.get_logs(path)
        return log
