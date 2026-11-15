"""Numerical recurrence, gradient, causality, and forecasting shape checks."""
import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from models import selective_scan, MambaBlock, Forecaster


class ModelTests(unittest.TestCase):
    def test_scan_against_closed_form_and_gradients(self):
        torch.manual_seed(0)
        u = torch.randn(2, 5, 3, dtype=torch.double, requires_grad=True)
        dt = torch.rand_like(u) * .1
        A = -torch.rand(3, 2, dtype=torch.double)
        B, C = torch.randn(2,5,2,dtype=torch.double), torch.randn(2,5,2,dtype=torch.double)
        D = torch.ones(3,dtype=torch.double)
        actual = selective_scan(u,dt,A,B,C,D)
        expected=[]
        for t in range(5):
            terms=[]
            for s in range(t+1):
                decay=(dt[:,s+1:t+1].sum(1).unsqueeze(-1)*A).exp()
                terms.append(decay*dt[:,s,:,None]*B[:,s,None,:]*u[:,s,:,None])
            expected.append((sum(terms)*C[:,t,None,:]).sum(-1)+u[:,t]*D)
        expected=torch.stack(expected,1)
        torch.testing.assert_close(actual,expected)
        ga=torch.autograd.grad(actual.square().sum(),u,retain_graph=True)[0]
        gb=torch.autograd.grad(expected.square().sum(),u)[0]
        torch.testing.assert_close(ga,gb)

    def test_mamba_causal(self):
        m=MambaBlock(8,4).eval()
        x=torch.randn(2,12,8); changed=x.clone(); changed[:,6:]+=10
        torch.testing.assert_close(m(x)[:,:6],m(changed)[:,:6])

    def test_models_backward(self):
        cfg=dict(width=8,history=96,horizon=96,state_size=4)
        for family in ['cnn','gru','transformer','mamba']:
            m=Forecaster(family,5,cfg)
            y=m(torch.randn(2,96,5))
            self.assertEqual(y.shape,(2,96,5))
            y.square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
