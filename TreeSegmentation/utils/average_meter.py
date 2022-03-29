class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, max_len=-1):
        self.vals = []
        self.counts = []
        self.max_len = max_len
        self.avg = 0

    def update(self, val, n=1):
        self.vals.append(val * n)
        self.counts.append(n)
        if self.max_len > 0 and len(self.vals) > self.max_len:
            self.vals = self.vals[-self.max_len:]
            self.counts = self.counts[-self.max_len:]
        self.avg = sum(self.vals) / sum(self.counts)