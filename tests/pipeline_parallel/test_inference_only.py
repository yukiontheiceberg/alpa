import unittest

import jax
import jax.numpy as jnp
import numpy as np

from alpa import (init, shutdown, parallelize, PipeshardParallel,
                  mark_pipeline_boundary)
from alpa.model.bert_model import BertConfig, FlaxBertLayerCollection
from alpa.testing import (MLPModel, create_train_state, mlp_inference_step,
                          bert_layer_collection_inference_step, assert_allclose)
from alpa.pipeline_parallel.layer_construction import (LayerOption,
                                                       AutoLayerOption,
                                                       ManualLayerOption)
from alpa.pipeline_parallel.stage_construction import (StageOption,
                                                       AutoStageOption,
                                                       UniformStageOption)
from alpa.shard_parallel.auto_sharding import AutoShardingOption, LogicalDeviceMesh

class PipelineInferenceTest(unittest.TestCase):

    def setUp(self):
        init(cluster="ray")

    # pylint: disable=no-self-use
    def tearDown(self):
        shutdown()

    def run_mlp_inference(self, manual_pipeline_layer, num_micro_batches, batch_size, hidden_size, num_layers):
        method = PipeshardParallel(num_micro_batches=8,
                                   pipeline_schedule="inference",
                                   layer_option=#"manual")
                                    ManualLayerOption(
                                    remat_layer=False) if manual_pipeline_layer else
                                    AutoLayerOption(layer_num=2, remat_layer=False))

        # Init model and optimizer
        # batch_size = 32
        # hidden_size = 16
        # num_layers = 10

        model = MLPModel(hidden_size=hidden_size,
                         num_layers=num_layers,
                         add_manual_pipeline_marker=manual_pipeline_layer)
        rngkey = jax.random.PRNGKey(0)
        x = jax.random.normal(rngkey, (batch_size, hidden_size), jnp.float32)
        y = jax.random.normal(rngkey, (batch_size, hidden_size), jnp.float32)
        batch = {'x': x, 'y': y}
        state = create_train_state(rngkey, model, [x])

        # Compile
        serial_inference_step = mlp_inference_step

        parallel_inference_step = parallelize(mlp_inference_step,
                                              method=method,
                                              donate_argnums=())
        executable = parallel_inference_step.get_executable(state, batch)

        # Run correctnesss test
        serial_out = serial_inference_step(state, batch)
        parallel_out = parallel_inference_step(state, batch)
        assert_allclose(serial_out, parallel_out, 1e-3, 1e-3)

        hlo_text = executable.get_hlo_text()
        return hlo_text

    # def run_bert_layer_collection_inference(self, manual_pipeline_layer):
    #     method = PipeshardParallel(num_micro_batches=4,
    #                                pipeline_schedule="inference",
    #                                layer_option="manual")

    #     # Init model and optimizer
    #     batch_size = 16
    #     seq_len = 256
    #     hidden_size = 512
    #     num_heads = 512 // 64
    #     n_layers = 2

    #     model = FlaxBertLayerCollection(
    #         config=BertConfig(hidden_size=hidden_size,
    #                           intermediate_size=hidden_size * 4,
    #                           num_attention_heads=num_heads,
    #                           num_hidden_layers=n_layers,
    #                           add_manual_pipeline_markers=manual_pipeline_layer,
    #                           pipeline_mp_size=n_layers))
    #     rngkey = jax.random.PRNGKey(0)
    #     x = jax.random.normal(rngkey, (batch_size, seq_len, hidden_size),
    #                           dtype=jnp.float32)
    #     y = jax.random.normal(rngkey, (batch_size, seq_len, hidden_size),
    #                           dtype=jnp.float32)
    #     attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.float32)
    #     batch = {"x": x, "y": y, "attention_mask": attention_mask}
    #     state = create_train_state(rngkey, model, [x, attention_mask])

    #     # Compile
    #     serial_inference_step = bert_layer_collection_inference_step
    #     parallel_inference_step = parallelize(
    #         bert_layer_collection_inference_step,
    #         method=method,
    #         donate_argnums=())
    #     executable = parallel_inference_step.get_executable(state, batch)
    #     executable = parallel_inference_step.get_executable(state, batch)

    #     # Run correctnesss test
    #     serial_out = serial_inference_step(state, batch)
    #     parallel_out = parallel_inference_step(state, batch)
    #     assert_allclose(serial_out, parallel_out, 1e-3, 1e-3)

    #     hlo_text = executable.get_hlo_text()
    #     return hlo_text

    def test_mlp1(self):
        self.run_mlp_inference(manual_pipeline_layer=True, num_micro_batches=4, batch_size=8, hidden_size=32, num_layers=2)
        # assert 1==0

    # def test_mlp2(self):
    #     self.run_mlp_inference(manual_pipeline_layer=True, num_micro_batches=4, batch_size=16, hidden_size=8, num_layers=4)

    # def test_mlp3(self):
    #     self.run_mlp_inference(manual_pipeline_layer=True, num_micro_batches=4, batch_size=32, hidden_size=8, num_layers=8)

    # def test_mlp4(self):
    #     self.run_mlp_inference(manual_pipeline_layer=True, num_micro_batches=4, batch_size=32, hidden_size=16, num_layers=2)

    # def test_mlp5(self):
    #     self.run_mlp_inference(manual_pipeline_layer=True, num_micro_batches=8, batch_size=32, hidden_size=8, num_layers=8)

    # def test_mlp6(self):
    #     self.run_mlp_inference(manual_pipeline_layer=True, num_micro_batches=8, batch_size=32, hidden_size=16, num_layers=4)

    # def test_mlp7(self):
    #     self.run_mlp_inference(manual_pipeline_layer=True, num_micro_batches=8, batch_size=64, hidden_size=16, num_layers=4)

    # def test_mlp8(self):
    #     self.run_mlp_inference(manual_pipeline_layer=True, num_micro_batches=2, batch_size=16, hidden_size=16, num_layers=4)

    # def test_mlp9(self):
    #     self.run_mlp_inference(manual_pipeline_layer=True, num_micro_batches=2, batch_size=32, hidden_size=8, num_layers=2)

    # def test_mlp10(self):
    #     self.run_mlp_inference(manual_pipeline_layer=True, num_micro_batches=2, batch_size=64, hidden_size=16, num_layers=8)

    # def test_mlp11(self):
    #     self.run_mlp_inference(manual_pipeline_layer=True, num_micro_batches=16, batch_size=64, hidden_size=32, num_layers=4)


    # def test_bert(self):
    #     self.run_bert_layer_collection_inference(True)

    # def test_output(self):
    #     method = PipeshardParallel(num_micro_batches=1,
    #                                pipeline_schedule="inference",
    #                                layer_option="manual")

    #     @parallelize(method=method)
    #     def func():
    #         a = jnp.ones(32)
    #         mark_pipeline_boundary()
    #         b = jnp.ones(32) * 2
    #         return a, b, 3

    #     a, b, c = func()

    #     assert_allclose(a, np.ones(32))
    #     assert_allclose(b, np.ones(32) * 2)
    #     assert_allclose(c, 3)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(PipelineInferenceTest("test_mlp1"))
    # suite.addTest(PipelineInferenceTest("test_mlp2"))
    # suite.addTest(PipelineInferenceTest("test_mlp3"))
    # suite.addTest(PipelineInferenceTest("test_mlp4"))
    # suite.addTest(PipelineInferenceTest("test_mlp5"))
    # suite.addTest(PipelineInferenceTest("test_mlp6"))
    # suite.addTest(PipelineInferenceTest("test_mlp7"))
    # suite.addTest(PipelineInferenceTest("test_mlp8"))
    # suite.addTest(PipelineInferenceTest("test_mlp9"))    
    # suite.addTest(PipelineInferenceTest("test_mlp10"))
    # suite.addTest(PipelineInferenceTest("test_mlp11"))
    # suite.addTest(PipelineInferenceTest("test_bert"))
    # suite.addTest(PipelineInferenceTest("test_output"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
