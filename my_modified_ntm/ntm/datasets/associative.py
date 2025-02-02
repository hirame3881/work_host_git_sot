import torch
from torch.utils.data import Dataset
from torch.distributions.binomial import Binomial


class AssociativeDataset(Dataset):
    """A Dataset class to generate random examples for associative recall task.

    Each input consists of a list of items with the number of itmes between
    `min_item` and `max_item`. An item is a sequence of binary vectors bounded
    on left and right by delimiter symbols. The list is followed by query item
    selected randomly from input items. It too is bounded by query delimiters.

    Target returns the item next to the query item.
    """

    def __init__(self, task_params):
        """Initialize a dataset instance for Associative Recall task.

        Arguments
        ---------
        task_params : dict
            A dict containing parameters relevant to associative recall task.
        """
        self.seq_width = task_params["seq_width"]
        self.seq_len = task_params["seq_len"]
        self.min_item = task_params["min_item"]
        self.max_item = task_params["max_item"]
        self.numitem="none"

    def __len__(self):
        # sequences are generated randomly so this does not matter
        # set a sufficiently large size for data loader to sample mini-batches
        return 65536
    def set_numitem(self):
        self.numitem = torch.randint(
            self.min_item, self.max_item, (1,), dtype=torch.long).item()
    def __getitem__(self, idx):
        # idx only acts as a counter while generating batches.
        num_item=self.numitem

        prob = 0.5 * \
            torch.ones([self.seq_len, self.seq_width], dtype=torch.float64)
        seq = Binomial(1, prob)

        # fill in input two bit wider than target to account for delimiter
        # flags.
        input_items = torch.zeros(
            [(self.seq_len + 1) * (num_item + 1) + 1, self.seq_width + 2])
        for i in range(num_item):
            input_items[(self.seq_len + 1) * i, self.seq_width] = 1.0
            input_items[(self.seq_len + 1) * i + 1:(self.seq_len + 1)
                        * (i + 1), :self.seq_width] = seq.sample()

        # generate query item randomly
        # in case of only one item, torch.randint throws error as num_item-1=0
        query_item = 0
        if num_item != 1:
            query_item = torch.randint(
                0, num_item - 1, (1,), dtype=torch.long).item()
        query_seq = input_items[(self.seq_len + 1) * query_item +
                                1:(self.seq_len + 1) * (query_item + 1), :self.seq_width]
        input_items[(self.seq_len + 1) * num_item,
                    self.seq_width + 1] = 1.0  # query delimiter
        input_items[(self.seq_len + 1) * num_item + 1:(self.seq_len + 1)
                    * (num_item + 1), :self.seq_width] = query_seq
        input_items[(self.seq_len + 1) * (num_item + 1),
                    self.seq_width + 1] = 1.0  # query delimiter

        # generate target sequences(item next to query in the input list)
        target_item = torch.zeros([self.seq_len, self.seq_width])
        # in case of last item, target sequence is zero
        if query_item != num_item - 1:
            target_item[:self.seq_len, :self.seq_width] = input_items[
                (self.seq_len + 1) * (query_item + 1) + 1:(self.seq_len + 1) * (query_item + 2), :self.seq_width]

        return {'input': input_items, 'target': target_item}
