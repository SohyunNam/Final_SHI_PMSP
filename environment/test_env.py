import json, os
from environment.test_simulation import *

weight = {80: {"ATCS": [5.88, 0.95], "COVERT": 6.605},
          160: {"ATCS": [6.605, 0.874], "COVERT": 1.4},
          240: {"ATCS": [7.053, 0.848], "COVERT": 0.9}}


def test(num_block=80, num_line=3, block_sample=None, sample_data=None, routing_rule=None, file_path=None, ddt=None, pt_var=None, test_num=50):
    tard_list = list()
    setup_list = list()

    for episode in range(test_num):
        block_list = sample_data["block_list"]
        due_date_list = sample_data["due_date"]
        iat = (960 * 6 * round(num_block / 80)) / num_block
        # if num_block == 240:
        #     week3_due_date = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19]
        #     due_date_list = np.random.choice(week3_due_date, size=num_block)
        #     # iat = (960 * 18) / num_block
        # elif num_block == 160:
        #     week2_due_date = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
        #     due_date_list = np.random.choice(week2_due_date, size=num_block)
        #     # iat = (960 * 12) / num_block
        # else:
        #     due_date_list = list(np.random.randint(low=0, high=6, size=num_block))  # Week 1 버전
        #     # iat = (960 * 6) / num_block

        # simulation object modeling
        model = dict()
        env = simpy.Environment()
        monitor = Monitor()
        monitor.reset()

        sim_block = dict()
        num_jobs = 0
        create_dict = dict()
        # Steel class로 모델링 + self.sim_block에 블록 저장 + self.num_jobs 계산
        for block_idx in range(len(block_list)):
            block_name = block_list[block_idx]
            # block_due_date = due_date_list[block_idx]
            block_data = block_sample[block_name]
            total_pt = 0.0
            for steel_name in block_data.keys():
                avg_speed = 1200 - ((block_data[steel_name]["weld_size"] - 4.5) / 0.5) * 50
                total_pt += (block_data[steel_name]["length"] / avg_speed) * block_data[steel_name]["num_steel"]

            create_time = due_date_list[block_idx]
            block_due_date = math.floor(create_time + ((total_pt * ddt) / (24 * 60)))
            sim_block["Block_{0}".format(block_idx)] = dict()
            sim_block["Block_{0}".format(block_idx)]["due_date"] = block_due_date
            sim_block["Block_{0}".format(block_idx)]["num_steel"] = 0

            steel_idx = 0
            block_steel_list = list()
            for steel_name in block_sample[block_name].keys():
                for i in range(block_sample[block_name][steel_name]["num_steel"]):
                    sim_block["Block_{0}".format(block_idx)][
                        "Steel_{0}_{1}_{2}".format(block_idx, steel_idx, i)] = Steel(
                        name="Steel_{0}_{1}_{2}".format(block_idx, steel_idx, i),
                        block="Block_{0}".format(block_idx), steel="Steel_{0}_{1}".format(block_idx, steel_idx),
                        feature=block_sample[block_name][steel_name], due_date=block_due_date)

                    block_steel_list.append(copy.deepcopy(sim_block["Block_{0}".format(block_idx)][
                                                              "Steel_{0}_{1}_{2}".format(block_idx, steel_idx, i)]))
                    sim_block["Block_{0}".format(block_idx)]["num_steel"] += 1
                    num_jobs += 1
                steel_idx += 1
            if create_time not in create_dict.keys():
                create_dict[create_time] = list()
            create_dict[create_time].append(copy.deepcopy(block_steel_list))

        routing = Routing(env, model, sim_block, monitor, num_jobs, routing_rule=routing_rule)
        model["Source"] = Source(env, create_dict, routing, iat, monitor)

        for i in range(num_line):
            model["Line {0}".format(i)] = Process(env, "Line {0}".format(i), model, routing, monitor, pt_var=pt_var)
            model["Line {0}".format(i)].reset()
        model["Sink"] = Sink(env, sim_block, monitor)
        model["Sink"].reset()

        env.run()
        # monitor.get_logs(file_path=file_path + '_{0}.csv'.format(episode))

        tard_list.append(sum(monitor.tardiness) / num_block)
        setup_list.append(monitor.setup / model["Sink"].total_finish)

    return np.mean(tard_list), np.mean(setup_list)