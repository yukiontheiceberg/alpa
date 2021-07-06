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
                      mesh_shape, mesh_mapping,
                      embedding_spec, indices_spec, output_spec):
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
        with mesh(mesh_devices, (mesh_mapping)):
            actual = parallel_func(embedding, indices)

        expected = func(embedding, indices)
        assert_allclose(np.array(actual), np.array(expected))

    def run_scatter_2d(self, vocab_size, hidden_size, batch_size, seq_len,
                       mesh_shape, mesh_mapping,
                       embedding_spec, indices_spec, updates_spec):
        def func(embedding, indices, updates):
            dimension_numbers = jax.lax.ScatterDimensionNumbers(
                update_window_dims=(2,),
                inserted_window_dims=(0,),
                scatter_dims_to_operand_dims=(0,),
            )
            ret = jax.lax.scatter_add(embedding, indices, updates, dimension_numbers)
            return ret

        parallel_func = pjit(
            func,
            in_axis_resources=(embedding_spec, indices_spec, updates_spec),
            out_axis_resources=embedding_spec)

        embedding = np.zeros((vocab_size, hidden_size), dtype=np.float32)
        indices = np.random.randint(0, vocab_size, (batch_size, seq_len, 1), np.int32)
        updates = np.arange(batch_size * seq_len * hidden_size).\
                reshape(batch_size, seq_len, hidden_size).astype(np.float32)

        mesh_devices = np.array(jax.devices()[:np.prod(mesh_shape)]).reshape(mesh_shape)
        with mesh(mesh_devices, (mesh_mapping)):
            actual = parallel_func(embedding, indices, updates)

        expected = func(embedding, indices, updates)
        print(self.hlo_module.to_string())
        assert_allclose(np.array(actual), np.array(expected))

    def test_gather_2d_partition_SR_RS_SS(self):
        # 2d partition, indices x operand = output (S,R x R,S = S,S)
        vocab_size = 512
        hidden_size = 32
        batch_size = 8
        seq_len = 16

        embedding_spec = P(None, 'y')
        indices_spec = P('x', None)
        output_spec= P('x', None, 'y')
        for mesh_shape in [(2, 2), (1, 4), (4, 1)]:
            for mesh_mapping in [('x', 'y'), ('y', 'x')]:
                self.run_gather_2d(vocab_size, hidden_size, batch_size, seq_len,
                                   mesh_shape, mesh_mapping,
                                   embedding_spec, indices_spec, output_spec)
                # No communication is required
                assert "channel_id" not in self.hlo_module.to_string()
                return

    def test_gather_2d_partition_SS_SR_SR(self):
        # 2d partition, indices x operand = output (S,S x S,R = S,R)
        vocab_size = 512
        hidden_size = 32
        batch_size = 8
        seq_len = 16

        embedding_spec = P('y', None)
        indices_spec = P('x', None)
        output_spec= P('x', None, None)
        for mesh_shape in [(2, 2), (1, 4), (4, 1)]:
            for mesh_mapping in [('x', 'y'), ('y', 'x')]:
                self.run_gather_2d(vocab_size, hidden_size, batch_size, seq_len,
                                   mesh_shape, mesh_mapping,
                                   embedding_spec, indices_spec, output_spec)
                hlo_ir = self.hlo_module.to_string()
                if "channel_id" in hlo_ir:
                    # Can have at most one all-reduce
                    assert hlo_ir.count("channel_id") == 1
                    assert hlo_ir.count("all-reduce(") == 1

    def test_scatter_2d_partition_SR_RS_SS(self):
        # 2d partition, indices x operand = updates (S,R x R,S = S,S)
        vocab_size = 512
        hidden_size = 32
        batch_size = 8
        seq_len = 16

        embedding_spec = P(None, None)
        indices_spec = P('x', None)
        updates_spec = P('x', None, None)
        for mesh_shape in [(4,)]:
            for mesh_mapping in [('x',)]:
                self.run_scatter_2d(vocab_size, hidden_size, batch_size, seq_len,
                                    mesh_shape, mesh_mapping,
                                    embedding_spec, indices_spec, updates_spec)

                # No communication is required
                #assert "channel_id" not in self.hlo_module.to_string()
                return

        #embedding_spec = P(None, 'y')
        #indices_spec = P('x', None)
        #updates_spec= P('x', None, 'y')
        #for mesh_shape in [(2, 2), (1, 4), (4, 1)]:
        #    for mesh_mapping in [('x', 'y'), ('y', 'x')]:
        #        self.run_scatter_2d(vocab_size, hidden_size, batch_size, seq_len,
        #                            mesh_shape, mesh_mapping,
        #                            embedding_spec, indices_spec, updates_spec)

        #        # No communication is required
        #        #assert "channel_id" not in self.hlo_module.to_string()
        #        return


def suite():
    suite = unittest.TestSuite()
    suite.addTest(SPMDPartitionerTest("test_gather_2d_partition_SR_RS_SS"))
    suite.addTest(SPMDPartitionerTest("test_gather_2d_partition_SS_SR_SR"))

    #suite.addTest(SPMDPartitionerTest("test_scatter_2d_partition_SR_RS_SS"))

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())

