import numpy as np
import torch
import torch.utils.data

from transformer import Constants


class EventData(torch.utils.data.Dataset):
    """ Event stream dataset. """
        
    # def __init__(self, data):
    #     """
    #     Data should be a list of event streams; each event stream is a list of dictionaries;
    #     each dictionary contains: time_since_start, time_since_last_event, type_event
    #     """
    #     self.time = [[elem['time_since_start'] for elem in inst] for inst in data]
    #     self.time_gap = [[elem['time_since_last_event'] for elem in inst] for inst in data]
    #     # plus 1 since there could be event type 0, but we use 0 as padding
    #     self.event_type = [[elem['type_event'] + 1 for elem in inst] for inst in data]
    #
    #     self.length = len(data)

    def __init__(self, data):
        """
        Data should be a list of event streams; each event stream is a list of dictionaries;
        each dictionary contains: time_since_start, time_since_last_event, type_event
        """
        self.time = [[elem['time_since_start'] for elem in inst['actions']] for inst in data]
        self.time_gap = [[elem['time_since_last_event'] for elem in inst['actions']] for inst in data]
        # plus 1 since there could be event type 0, but we use 0 as padding
        self.event_type = [[elem['type_event'] + 1 for elem in inst['actions']] for inst in data]
        # self.event_user_group = [[elem['event_user_group'] for elem in inst['actions']] for inst in data]

        self.label = [[elem + 1 for elem in inst['label']] for inst in data]
        self.group_ = [[elem + 1 for elem in inst['group_']] for inst in data]
        self.distance = [[elem for elem in inst['distance']] for inst in data]
        self.track = [[elem for elem in inst['track']] for inst in data]


        # self.event_len = [inst['event_len'] for inst in data]
        # self.in_out = [inst['in_out'] for inst in data]
        #
        # self.distance = [inst['distance'] for inst in data]

        self.length = len(data)

    def __len__(self):
        return self.length

        # self.event_user_group[idx],
    def __getitem__(self, idx):
        """ Each returned element is a list, which represents an event stream """
        # return self.time[idx], self.time_gap[idx], self.event_type[idx], self.in_out[idx], self.distance[idx]
        return self.time[idx], self.time_gap[idx], self.event_type[idx], \
               self.label[idx], self.group_[idx], self.distance[idx], self.track[idx]


def pad_time(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst[0]) for inst in insts)

    batch_seq = np.array([
        inst[0][:max_len] + [Constants.PAD] * (max_len - len(inst[0]))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.float32)


def pad_type(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst[0]) for inst in insts)
    # max_len = 128  # max(len(inst[0]) for inst in insts)
    batch_seq = np.array([
        inst[0][:max_len] + [Constants.PAD] * (max_len - len(inst[0]))
        for inst in insts])
    return torch.tensor(batch_seq, dtype=torch.long)


def pad_scores(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst[0]) for inst in insts)

    batch_seq = np.array([
        inst[0][:max_len] + [Constants.PAD] * (max_len - len(inst[0]))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.float32)


def padding_event_label(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst[0]) for inst in insts)

    batch_seq = np.array([
        inst[0][:max_len] + [Constants.PAD] * (max_len - len(inst[0]))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.long)


def pad_where(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst[0]) for inst in insts)

    batch_seq = np.array([
        inst[0] + [Constants.PAD] * (max_len - len(inst[0]))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.long)


def pad_distance(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst[0]) for inst in insts)

    batch_seq = np.array([
        inst[0] + [Constants.PAD] * (max_len - len(inst[0]))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.float32)


def pad_group(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst[0]) for inst in insts)

    batch_seq = np.array([
        inst[0][:max_len] + [Constants.PAD] * (max_len - len(inst[0]))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.long)


# def pad_coor(insts):
#     """ Pad the instance to the max seq length in batch. """
#
#     max_len = max(len(inst[0]) for inst in insts)
#
#     # print(insts)
#     batch_seq = np.array([
#         inst[0] + [] * (max_len - len(inst[0]))
#         for inst in insts])
#
#     return torch.tensor(batch_seq, dtype=torch.float32)


def padding_(insts):  # (16, L)
    """ Pad the instance to the max seq length in batch. """
    # print(insts)
    max_len = max(len(inst[0]) for inst in insts)
    inner_dis = []

    for i,io in enumerate(insts):
        len_ = max_len - len(io[0])

        if len(io[0])>max_len:
            len_ = 0

        # print(io[0].size())

        pad_width1 = ((0, len_), (0, len_))
        inner_dis.append(np.pad(io[0], pad_width=pad_width1, mode='constant', constant_values=0))  # [:max_len,:max_len]

    return torch.tensor(inner_dis, dtype=torch.float32)


def collate_fn(insts):
    """ Collate function, as required by PyTorch. """

    ds = insts
    # print(len(ds),type(ds), type(ds[0]))
    # print(len(ds),type(ds), type(ds[0]))
    (event_type, score, lats, lngs, test_label, test_score, inner_dis, inner_poi_sim, group_) = list(zip(*ds))
    # time = pad_time(time)
    # time_gap = pad_time(time_gap)
    event_type = pad_type(event_type)
    score = pad_scores(score)
    # event_user_group = padding_event_user_group(event_user_group)
    test_label = padding_event_label(test_label)
    test_score = pad_scores(test_score)
    # lats = pad_distance(lats)
    # lngs = pad_distance(lngs)
    #
    # # group_ = pad_group(group_)
    # # print(group_)
    # # print(inner_dis)
    inner_dis = padding_(inner_dis)
    # inner_poi_sim = padding_(inner_poi_sim)
    # print(inner_dis)
    # print(group_)
    return event_type, score, torch.tensor([], dtype=torch.float32), torch.tensor([], dtype=torch.float32), test_label, test_score, inner_dis.clone().detach()
           # inner_dis.clone().detach(), group_
           # torch.tensor(inner_dis, dtype=torch.float32), torch.tensor(insts[0][1], dtype=torch.float32), group_



def get_dataloader(data, batch_size, shuffle=True):
    """ Prepare dataloader. """

    # ds = EventData(data)
    ds = data
    # for i in ds:
    #     if len(i)==0:
    #         ds.remove(i)
    # print(len(data))
    # print(data[0])
    # print(ds[0])
    # ([100.64770361111111, 100.647495, 116.84664333333333, 109.98184611111111, 111.50314083333333, 88.43541916666666,
    #   126.49437833333333, 106.02323194444445, 122.71028111111112, 111.85150194444445, 99.645685, 97.72648694444445,
    #   112.68185138888889, 98.51896138888888, 105.69572722222222, 119.89840083333333, 129.66919444444446,
    #   99.0735211111111, 129.4669638888889, 99.33444777777778, 100.64828888888889, 110.18111222222223,
    #   110.23011055555556, 86.4169613888889, 109.98181555555556, 103.07077027777778, 104.97431777777778, 128.7920425,
    #   128.83506694444443, 82.9105888888889, 109.98174527777778, 99.64573111111112, 114.61724833333334,
    #   129.3545777777778, 127.75363638888889, 100.64795722222222, 100.64763333333333, 107.657655],
    #  [0, 3419.0, 2410.0, 1170.0, 4519.0, 891.0, 4217.0, 143.0, 244.0, 143.0, 3583.0, 3914.0, 4660.0, 4552.0, 106.0,
    #   2985.0, 448.0, 5079.0, 1146.0, 3062.0, 1146.0, 4651.0, 4651.0, 3207.0, 338.0, 3830.0, 3704.0, 3704.0, 1276.0,
    #   1276.0, 4406.0, 2714.0, 2091.0, 2325.0, 2605.0, 2326.0, 2326.0, 2788.0],
    #  [3420, 2411, 1171, 4520, 892, 4218, 144, 245, 144, 3584, 3915, 4661, 4553, 107, 2986, 449, 5080, 1147, 3063, 1147,
    #   4652, 4652, 3208, 339, 3831, 3705, 3705, 1277, 1277, 4407, 2715, 2092, 2326, 2606, 2327, 2327, 2789, 3634],
    #  [1352, 3446, 793, 3250, 5215, 5105, 1599, 3877, 2149, 348], [], [], [])

    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=2,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle
    )
    return dl
