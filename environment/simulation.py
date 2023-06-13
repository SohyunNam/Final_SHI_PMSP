import simpy, math, copy, random
import pandas as pd
import numpy as np

np.random.seed(42)
random.seed(42)

class Steel:
    def __init__(self, name, block, steel, feature, due_date):
        # 해당 job의 이름
        self.name = name
        self.block = block
        self.steel = steel
        self.length = feature["length"]
        self.due_date = due_date
        #
        # self.web_face = [feature["web_width"], feature["web_thickness"], feature["face_width"],
        #                  feature["face_thickness"]]
        self.web_face = feature["feature"]  # 0 ~ 200 사이의 정수
        self.avg_speed = 1200 - ((feature["weld_size"] - 4.5) / 0.5) * 50
        self.avg_pt = feature["length"] / self.avg_speed

        self.started = 0.0
        self.completed = 0  # 종료 시간


class Source:
    def __init__(self, env, creating_dict, routing, iat, monitor):
        self.env = env
        self.name = "Source"
        self.routing = routing
        self.iat = iat
        self.monitor = monitor
        self.creating_list = list()
        day_list = list(creating_dict.keys())
        day_list.sort()
        for day in day_list:
            self.creating_list += creating_dict[day]

        # self.queue = simpy.FilterStore(env)
        env.process(self.run())

    def run(self):
        while len(self.creating_list):
            created_block = self.creating_list.pop(0)
            self.monitor.record(time=self.env.now, event="Block Created", block=created_block[0].block, process="Source", memo=created_block[0].due_date)
            for steel in created_block:
                self.routing.queue.put(steel)
            if self.routing.is_queue_event:
                self.routing.queue_event.succeed()
            self.routing.queue_list = copy.deepcopy([job.name for job in self.routing.queue.items])

            iat = np.random.exponential(self.iat)
            if (self.env.now + iat) % 1440 > 960:
                self.monitor.record(time=self.env.now, event="Day Off", process="Source")
                today = math.floor(self.env.now / 1440)
                next_day = today + 1 if today % 7 != 5 else today + 2
                to_next_day = next_day * 1440 - self.env.now
                yield self.env.timeout(to_next_day)
                self.monitor.record(time=self.env.now, event="Day On", process="Source")
            yield self.env.timeout(iat)


class Process:
    def __init__(self, env, name, model, routing, monitor, pt_var):
        self.env = env
        self.name = name
        self.model = model
        self.routing = routing
        self.monitor = monitor

        self.queue = simpy.Store(env)
        self.idle = True
        self.job = None
        self.block = None
        self.start_time = 0
        self.planned_finish_time = 0
        self.setup = random.randint(0, 200)
        self.pt_var = pt_var

        env.process(self.run())

    def run(self):
        while True:
            # if (len(self.queue.items) == 0) and (len(self.model["Source"].queue.items) == 0):
            #     break

            # Job Calling Event 등록
            self.routing.waiting_event.put([self.env.event(), self.name])

            self.monitor.record(time=self.env.now, event="Call the Job", process=self.name,
                                memo=len(self.routing.queue.items))
            setup_time = 0
            setup_memo = None
            part = yield self.queue.get()
            self.job = part

            self.start_time = self.env.now
            if self.setup != self.job.web_face:
                setup_time = 5
                setup_memo = "{0} to {1}".format(self.setup, self.job.web_face)
                self.monitor.setup += 1
                self.monitor.setup_list.append(1)
                self.setup = part.web_face
            else:
                self.monitor.record(time=self.env.now, part=part.name, block=part.block, event="Non-SetUp",
                                    process=self.name, memo="{0} to {1}".format(self.setup, self.job.web_face))

            self.block = part.block
            self.idle = False
            # 시작 시간 기록
            part.started = self.env.now
            # State 계산을 위한 예상 작업 시간 - 평균값 사용
            self.planned_finish_time = self.env.now + part.avg_pt + setup_time

            # 실제 작업 시간
            processing_time = np.random.uniform(low=part.avg_pt * (1 - self.pt_var), high=part.avg_pt * (1 + self.pt_var))

            # 다음 날로 이동
            if (self.env.now + setup_time + part.avg_pt) % 1440 > 960:
                self.monitor.record(time=self.env.now, event="Day Off", process=self.name)
                today = math.floor(self.env.now / 1440)
                next_day = today + 1 if today % 7 != 5 else today + 2
                to_next_day = next_day * 1440 - self.env.now
                yield self.env.timeout(to_next_day)
                self.monitor.record(time=self.env.now, event="Day On", process=self.name)

            # 셋업 발생
            if setup_memo is not None:
                self.monitor.record(time=self.env.now, part=part.name, block=part.block, event="Set-Up",
                                    process=self.name, memo=setup_memo)
                yield self.env.timeout(setup_time)
                self.setup = part.web_face  # 셋업 변경 일어남

            # 셋업 후 작업
            self.monitor.record(time=self.env.now, part=part.name, block=part.block, event="Work Start",
                                process=self.name, memo=processing_time)

            yield self.env.timeout(processing_time)  # 0.9 * 평균 ~ 1.1 * 평균 사이의 stochastic한 값
            self.monitor.record(time=self.env.now, part=part.name, block=part.block, event="Work Finish",
                                process=self.name)

            self.model["Sink"].put(part)
            self.idle = True

            if len(self.routing.queue.items) == 0 and self.routing.created == self.routing.end_num:
                break

    def reset(self):
        self.queue.items = list()
        self.idle = True
        self.job = None
        self.block = None
        self.start_time = 0
        self.setup = random.randint(0, 200)
        self.planned_finish_time = 0


class Routing:
    def __init__(self, env, model, block_info, monitor, end_num):
        self.env = env
        self.model = model
        self.block_info = block_info
        self.monitor = monitor
        self.end_num = end_num

        self.indicator = False
        self.decision = False
        self.routing_rule = None
        self.line = None

        self.idle = False
        self.job = None

        self.queue = simpy.FilterStore(env)
        self.waiting_event = simpy.Store(env)
        self.queue_list = list()
        # self.waiting_jobs = copy.deepcopy([job.name for job in self.model["Source"].queue.items])

        self.mapping = {0: "SSPT", 1: "ATCS", 2: "MDD", 3: "COVERT"}

        self.setup = False
        self.created = 0
        self.queue_event = self.env.event()
        self.is_queue_event = False
        env.process(self.run())

    def run(self):
        while self.created < self.end_num:
            if len(self.queue.items) == 0:
                self.is_queue_event = True
                yield self.queue_event
                self.is_queue_event = False
                self.queue_event = self.env.event()

            decision = yield self.waiting_event.get()
            self.decision = decision[0]
            location = decision[1]
            self.line = location
            self.indicator = True

            routing_rule = yield self.decision
            self.routing_rule = self.mapping[routing_rule]
            self.decision = None
            self.monitor.record(time=self.env.now, event="Routing Start", process=location, memo=self.routing_rule)
            self.setup = False

            if self.routing_rule == "SSPT":
                next_job = yield self.env.process(self.SSPT())
            elif self.routing_rule == "ATCS":
                next_job = yield self.env.process(self.ATCS())
            elif self.routing_rule == "MDD":
                next_job = yield self.env.process(self.MDD())
            elif self.routing_rule == "COVERT":
                next_job = yield self.env.process(self.COVERT())
            else:
                next_job = None

            self.created += 1
            self.monitor.record(time=self.env.now, part=next_job.name, block=next_job.block, event="Routing Finished",
                                process=self.line)

            if next_job.web_face != self.model[self.line].setup:
                self.setup = True
            self.model[self.line].queue.put(next_job)

    def SSPT(self):
        job_list = copy.deepcopy(self.queue_list)
        calling_line = self.model[self.line]
        idx_list = []
        for job_name in job_list:
            temp = job_name.split("_")
            job = self.block_info["Block_{0}".format(temp[1])][job_name]
            idx = job.avg_pt + self._get_setup_time(calling_line.job, job)
            idx_list.append(idx)

        job_idx = np.random.choice(np.where(idx_list == np.min(idx_list))[0])
        self.queue_list.remove(job_list[job_idx])
        next_job = yield self.queue.get(lambda x: x.name == job_list[job_idx])

        return next_job

    def ATCS(self):
        job_list = copy.deepcopy(self.queue_list)
        calling_line = self.model[self.line]
        k1 = 6.7
        k2 = 1.3
        p_avg = np.mean(
            [self.block_info["Block_{0}".format(job.split("_")[1])][job].avg_pt for job in job_list])
        s_avg = np.mean(
            [self._get_setup_time(calling_line.job, self.block_info["Block_{0}".format(job.split("_")[1])][job])
             for job in job_list])

        idx_list = []
        for job_name in job_list:
            temp = job_name.split("_")
            job = self.block_info["Block_{0}".format(temp[1])][job_name]
            second_exp = np.exp(-self._get_setup_time(calling_line.job, job) / (k2 * s_avg)) if s_avg != 0 else 1.0
            idx = (1 / job.avg_pt) * np.exp(
                -np.max((job.due_date * 1440 + 960) - job.avg_pt - self.env.now) / (k1 * p_avg)) * second_exp
            idx_list.append(idx)

        job_idx = np.random.choice(np.where(idx_list == np.max(idx_list))[0])
        self.queue_list.remove(job_list[job_idx])
        next_job = yield self.queue.get(lambda x: x.name == job_list[job_idx])

        return next_job

    def MDD(self):
        job_list = copy.deepcopy(self.queue_list)
        idx_list = []
        for job_name in job_list:
            temp = job_name.split("_")
            job = self.block_info["Block_{0}".format(temp[1])][job_name]
            idx = max(job.due_date * 1440 + 960, self.env.now + job.avg_pt)
            idx_list.append(idx)

        job_idx = np.random.choice(np.where(idx_list == np.min(idx_list))[0])
        self.queue_list.remove(job_list[job_idx])
        next_job = yield self.queue.get(lambda x: x.name == job_list[job_idx])

        return next_job

    def COVERT(self):
        job_list = copy.deepcopy(self.queue_list)
        K = 14

        idx_list = []
        for job_name in job_list:
            temp = job_name.split("_")
            job = self.block_info["Block_{0}".format(temp[1])][job_name]
            idx = (1 / job.avg_pt) * max(1 - (max(0, job.due_date * 1440 + 960 - job.avg_pt - self.env.now) )/ K * job.avg_pt, 0)
            idx_list.append(idx)

        job_idx = np.random.choice(np.where(idx_list == np.max(idx_list))[0])
        self.queue_list.remove(job_list[job_idx])
        next_job = yield self.queue.get(lambda x: x.name == job_list[job_idx])

        return next_job

    def _get_setup_time(self, job1, job2):
        setup_time = 0.0
        if (job1 is not None) and (job2 is not None) and (job1.web_face != job2.web_face):
            setup_time = 5.0
        return setup_time

    def reset(self):
        self.indicator = False
        self.decision = False
        self.routing_rule = None
        self.line = None

        self.idle = False
        self.job = None

        self.queue.items = list()
        self.waiting_event.items = list()
        self.queue_list = list()

        self.setup = False
        self.created = 0
        self.queue_event = self.env.event()
        self.is_queue_event = False


class Sink:
    def __init__(self, env, block_info, monitor):
        self.env = env
        self.block_info = block_info
        self.monitor = monitor

        self.finished = dict()
        self.finished_block = list()
        self.total_finish = 0
        self.makespan = 0.0

    def put(self, job):
        if job.block not in self.finished.keys():
            self.finished[job.block] = dict()
            self.finished[job.block]["num"] = 0
            self.finished[job.block]["time"] = [self.env.now]
            self.finished[job.block]["start"] = job.started

        self.finished[job.block]["num"] += 1  # jobtype 별 종료 개수
        self.total_finish += 1

        self.makespan = self.env.now / 1440

        job.completed = math.floor(self.env.now/1440)  # 끝난 날짜
        difference = max((self.env.now / 1440) - self.block_info[job.block]["due_date"] - 1, 0) # due date 대비 얼마나 늦게 끝났는 지 (현재 시각 - 납기일)
        self.finished_block.append(difference) # Steel 단위로 tardiness penalty 주는 걸로 바꿈, # State 및 Reward 시 사용

        day = math.floor(self.env.now / 1440)  # 현재 날짜
        if day not in self.monitor.throughput.keys():
            self.monitor.throughput[day] = 0
        self.monitor.throughput[day] += 1
        self.monitor.record(time=self.env.now, part=job.name, block=job.block, event="Completed", process="Sink",
                            memo=difference)

        if self.finished[job.block]["num"] == self.block_info[job.block]["num_steel"]:
            self.monitor.record(time=self.env.now, part=job.name, block=job.block, event="Block Completed",
                                process="Sink", memo=difference)
            self.monitor.tardiness.append(difference)

            self.finished[job.block]["time"].append(self.env.now)

    def reset(self):
        self.finished = dict()
        self.finished_block = list()
        self.total_finish = 0


class Monitor:
    def __init__(self):
        self.day = list()
        self.time = list()
        self.part = list()
        self.block = list()
        self.event = list()
        self.process = list()
        self.memo = list()

        self.setup = 0
        self.setup_list = list()
        self.tardiness = list()
        self.throughput = dict()

    def record(self, time=None, part=None, block=None, event=None, process=None, memo=None):
        self.time.append(round(time, 2))
        self.day.append(int(math.floor(time/1440)))
        self.part.append(part)
        self.block.append(block)
        self.event.append(event)
        self.process.append(process)
        self.memo.append(memo)

    def get_logs(self, file_path=None):
        event_tracer = pd.DataFrame(columns=["Day", "Part", "Block", "Event", "Process", "Memo", "Time"])
        event_tracer["Day"] = self.day
        event_tracer["Part"] = self.part
        event_tracer["Block"] = self.block
        event_tracer["Event"] = self.event
        event_tracer["Process"] = self.process
        event_tracer["Memo"] = self.memo
        event_tracer["Time"] = self.time
        if file_path is not None:
            event_tracer.to_csv(file_path, encoding='utf-8-sig', index=False)
            # print(file_path)

        return event_tracer
        # print(self.filepath)

    def reset(self):
        self.day = list()
        self.time = list()
        self.part = list()
        self.block = list()
        self.event = list()
        self.process = list()
        self.memo = list()

        self.setup = 0
        self.tardiness = list()


