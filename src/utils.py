import torch

def sinkhorn(self, Q, nmb_iters=3):
    with torch.no_grad():
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        K, B = Q.shape

        if self.gpus > 0:
            u = torch.zeros(K).cuda()
            r = torch.ones(K).cuda() / K
            c = torch.ones(B).cuda() / B
        else:
            u = torch.zeros(K)
            r = torch.ones(K) / K
            c = torch.ones(B) / B

        for _ in range(nmb_iters):
            u = torch.sum(Q, dim=1)

            Q *= (r / u).unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)

        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()