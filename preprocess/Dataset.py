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


def pad_aspect(insts):
    """ Pad the instance to the max seq length in batch. """

    res = []
    max_len = max(len(inst[0]) for inst in insts)

    # print([len(inst[0]) for inst in insts], max_len)
    for inst in insts:
        for i in range(max_len - len(inst[0])):
            # np.array([
            #         inst[0] + [[0, 0, 0, 0, 0, 0], ] * (max_len - len(inst[0]))])
            # print(inst[0])
            inst[0].append([0] * 62)
            # print(inst[0])

        # print([len(i) for i in inst[0]], max_len)

        m = np.array(inst[0])
        res.append(m)

    # print([len(inst[0]) for inst in res])
    # print('=======================')
    return torch.tensor(np.array(res), dtype=torch.float32)


def collate_fn(insts):
    """ Collate function, as required by PyTorch. """

    ds = insts
    # print(ds[0])
    (event_type, score, time, aspect, test_label, test_score, inner_dis) = list(zip(*ds))
    time = pad_time(time)
    aspect = pad_aspect(aspect)
    # time_gap = pad_time(time_gap)
    event_type = pad_type(event_type)
    test_label = padding_event_label(test_label)

    score = pad_scores(score)
    test_score = pad_scores(test_score)

    inner_dis = padding_(inner_dis)
    return event_type, score, time, aspect, test_label, test_score, inner_dis.clone().detach()


def get_dataloader(data, batch_size, shuffle=True):
    """ Prepare dataloader. """

    # ds = EventData(data)
    ds = data

    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=2,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle
    )
    return dl
