class DatasetExistsError(Exception):
    def __init__(self, dataset):
        self.dataset = dataset
        self.message = "Dataset " + self.dataset + " already exists"
        super().__init__(self.message)


class DatasetNotFoundError(Exception):
    def __init__(self, dataset):
        self.dataset = dataset
        self.message = "Dataset " + self.dataset + " Not Found"
        super().__init__(self.message)


class GroupExistsError(Exception):
    def __init__(self, group):
        self.group = group
        self.message = "Group " + self.group + " already exists"
        super().__init__(self.message)


class GroupNotFoundError(Exception):
    def __init__(self, group):
        self.group = group
        self.message = "Group " + self.group + " does not exist"
        super().__init__(self.message)


class ParentLinkError(Exception):
    def __init__(self, parent, link):
        self.parent = parent
        self.link = link
        self.message = (
            "Can not delete link " + self.parent + " is parent to " + self.link
        )
        super().__init__(self.message)


class IndexOutOfRangeError(Exception):
    def __init__(self, idx, max_idx):
        self.idx = idx
        self.max_idx = max_idx
        self.message = (
            "Chunk index: "
            + str(self.idx)
            + " is out of bounds. Max index is: "
            + str(self.max_idx)
        )
        super().__init__(self.message)


class ConnectionError(Exception):
    pass