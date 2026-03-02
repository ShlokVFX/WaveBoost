import numpy as np
from queue import Queue
import time


class Request:
    def __init__(self, seq_id, input_tokens, max_gen_len=10):
        self.seq_id = seq_id                # Unique request ID
        self.input_tokens = input_tokens    # Input token sequence
        self.generated_tokens = []          # Generated tokens
        self.max_gen_len = max_gen_len      # Maximum generation length
        self.completed = False              # Whether generation is finished

    def is_completed(self):
        """
        Check if generation is complete.
        Stops when:
        - explicitly marked completed
        - generated tokens reach max_gen_len
        """
        return self.completed or len(self.generated_tokens) >= self.max_gen_len


# ============================================================
# Continuous Batching Engine
# ============================================================

class ContinuousBatchingEngine:
    def __init__(self, max_batch_size=8):
        self.request_queue = Queue()        # Incoming request queue
        self.active_requests = []           # Active unfinished requests
        self.max_batch_size = max_batch_size

    def add_request(self, request):
        """Add new request to queue"""
        self.request_queue.put(request)

    def get_next_batch(self):
        """
        Dynamically build next batch:
        - Keep unfinished requests
        - Add new requests until max_batch_size
        """
        # Keep unfinished requests
        batch = [r for r in self.active_requests if not r.is_completed()]

        # Add new requests from queue
        while not self.request_queue.empty() and len(batch) < self.max_batch_size:
            new_req = self.request_queue.get()
            batch.append(new_req)

        self.active_requests = batch
        return batch if batch else None

    def decode_step(self, batch):
        """
        Simulate one decoding step
        (In real systems this is model forward pass)
        """
        for req in batch:
            next_token = np.random.randint(0, 1000)
            req.generated_tokens.append(next_token)

            # Randomly mark some requests as finished (simulate EOS)
            if np.random.random() < 0.2:
                req.completed = True

    def run(self):
        """Run continuous batching inference"""
        step = 0
        while True:
            batch = self.get_next_batch()

            if not batch:
                if self.request_queue.empty():
                    break
                continue

            print(f"\nStep {step}: Processing batch (size={len(batch)})")

            self.decode_step(batch)

            for req in batch:
                status = "Completed" if req.is_completed() else "Running"
                print(
                    f"Request {req.seq_id}: "
                    f"Generated={len(req.generated_tokens)} ({status})"
                )

            step += 1
            time.sleep(0.5)


# ============================================================
# Selective Batching Engine (Length-Aware)
# ============================================================

class SelectiveBatchingEngine(ContinuousBatchingEngine):
    def __init__(self, max_batch_size=8):
        super().__init__(max_batch_size)

    def group_by_length(self, batch):
        """
        Group sequences by total length
        (used for Attention layer to avoid padding waste)
        """
        groups = {}

        for req in batch:
            seq_len = len(req.input_tokens) + len(req.generated_tokens)

            if seq_len not in groups:
                groups[seq_len] = []

            groups[seq_len].append(req)

        return groups

    def attention_step(self, groups):
        """
        Simulate Attention layer computation.
        Same-length sequences processed together.
        """
        print("Attention Layer:")
        for seq_len, group in groups.items():
            print(
                f"  Processing length {seq_len} group "
                f"(size={len(group)})"
            )

    def ffn_step(self, batch):
        """
        Simulate FFN layer computation.
        All sequences merged for maximum parallelism.
        """
        print(f"FFN Layer: Processing all {len(batch)} sequences together")

    def decode_step(self, batch):
        """
        Selective batching decode step:
        1. Group by length for Attention
        2. Merge all for FFN
        3. Generate next token
        """
        groups = self.group_by_length(batch)

        self.attention_step(groups)
        self.ffn_step(batch)

        for req in batch:
            next_token = np.random.randint(0, 1000)
            req.generated_tokens.append(next_token)

            if np.random.random() < 0.2:
                req.completed = True


# ============================================================
# Experiment Runner
# ============================================================

def run_experiment():
    # Create test requests (different input lengths)
    requests = [
        Request(seq_id=1, input_tokens=[1, 2, 3], max_gen_len=5),
        Request(seq_id=2, input_tokens=[4, 5], max_gen_len=8),
        Request(seq_id=3, input_tokens=[6], max_gen_len=3),
        Request(seq_id=4, input_tokens=[7, 8, 9, 10], max_gen_len=6),
    ]

    print("=== Testing Continuous Batching ===")
    engine = ContinuousBatchingEngine(max_batch_size=3)
    for req in requests:
        engine.add_request(req)
    engine.run()

    # Reset request states
    for req in requests:
        req.generated_tokens = []
        req.completed = False

    print("\n=== Testing Selective Batching ===")
    engine = SelectiveBatchingEngine(max_batch_size=3)
    for req in requests:
        engine.add_request(req)
    engine.run()


if __name__ == "__main__":
    run_experiment()