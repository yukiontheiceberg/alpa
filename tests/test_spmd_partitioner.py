"""Test our modification of the SPMD partitioner"""
from functools import partial
import unittest

import numpy as np

import jax
import jax.numpy as jnp
from jax.experimental import PartitionSpec as P
from jax.experimental.maps import mesh
from jax.experimental.pjit import pjit, with_sharding_constraint, pjit_callable, pjit_p

from parax.testing import assert_allclose

class SPMDPartitionerTest(unittest.TestCase):
    def setUp(self):
        assert len(jax.local_devices()) >= 4

        # Monkey patch to get the HloModule from compiled binary
        self.hlo_module = None

        def _pjit_call_impl(*args, jaxpr,
                            in_axis_resources, out_axis_resources,
                            resource_env, donated_invars, name):
          compiled = pjit_callable(
              jaxpr, in_axis_resources, out_axis_resources,
              resource_env, donated_invars, name)
          self.hlo_module = compiled.args[0].hlo_modules()[0]
          return compiled(*args)

        pjit_p.def_impl(_pjit_call_impl)

    def run_gather_2d(self, vocab_size, hidden_size, batch_size, seq_len,
                      mesh_shape, embedding_spec, indices_spec, output_spec):
        def func(embedding, indices):
          ret = jnp.take(embedding, indices, axis=0)
          return ret

        parallel_func = pjit(
            func,
            in_axis_resources=(embedding_spec, indices_spec),
            out_axis_resources=output_spec)

        embedding = np.arange(vocab_size * hidden_size).\
                reshape(vocab_size, hidden_size).astype(np.float32)
        indices = np.random.randint(0, vocab_size, (batch_size, seq_len), np.int32)

        mesh_devices = np.array(jax.devices()[:np.prod(mesh_shape)]).reshape(mesh_shape)
        with mesh(mesh_devices, ('x', 'y')):
            actual = parallel_func(embedding, indices)

        expected = func(embedding, indices)
        assert_allclose(np.array(actual), np.array(expected))

    def test_gather_2d_partition_SR_RS_SS(self):
        # 2d partition, indices x operand = output (S,R x R,S = S,S)
        vocab_size = 256
        hidden_size = 64
        batch_size = 8
        seq_len = 16

        embedding_spec = P(None, 'y')
        indices_spec = P('x', None)
        output_spec= P('x', None, 'y')
        for mesh_shape in [(2, 2), (1, 4), (4, 1)]:
            self.run_gather_2d(vocab_size, hidden_size, batch_size, seq_len,
                               mesh_shape, embedding_spec, indices_spec, output_spec)
            assert "channel_id" not in self.hlo_module.to_string()

    def test_gather_2d_partition_SS_SR_SR(self):
        # 2d partition, indices x operand = output (S,S x S,R = S,R)
        vocab_size = 256
        hidden_size = 64
        batch_size = 8
        seq_len = 16

        mesh_shape = (2, 2)
        embedding_spec = P('y', None)
        indices_spec = P('x', None)
        output_spec= P('x', None, None)
        self.run_gather_2d(vocab_size, hidden_size, batch_size, seq_len,
                           mesh_shape, embedding_spec, indices_spec, output_spec)

        print(self.hlo_module.to_string())


def suite():
    suite = unittest.TestSuite()
    #suite.addTest(SPMDPartitionerTest("test_gather_2d_partition_SR_RS_SS"))
    suite.addTest(SPMDPartitionerTest("test_gather_2d_partition_SS_SR_SR"))

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())

