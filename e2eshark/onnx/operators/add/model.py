# run.pl creates runmodel.py by concatenating this file model.py
# and tools/stubs/onnxmodel.py
# Description: testing gemm
# See https://onnx.ai/onnx/intro/python.html for intro on creating
# onnx model using python onnx API
import numpy
import onnxruntime
from onnx import numpy_helper, TensorProto, save_model
from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info
from onnx.checker import check_model


# Create an input (ValueInfoProto)
X = make_tensor_value_info("X", TensorProto.FLOAT, [4, 5])
Y = make_tensor_value_info("Y", TensorProto.FLOAT, [4, 5])

# Create an output
Z = make_tensor_value_info("Z", TensorProto.FLOAT, [4, 5])

# Create a node (NodeProto)
addnode = make_node(
    "Add", ["X", "Y"], ["Z"], "addnode"  # node name  # inputs  # outputs
)

# Create the graph (GraphProto)
graph = make_graph(
    [addnode],
    "addgraph",
    [X, Y],
    [Z],
)

# Create the model (ModelProto)
onnx_model = make_model(graph)
onnx_model.opset_import[0].version = 19

# Save the model
# save_model(onnx_model, "model.onnx")
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# test_input and test_output must be numpy
# bfloat16 is not supported by onnxruntime and numpy
# case to/from fp32 to work around at various stages
# start an onnxrt session
session = onnxruntime.InferenceSession("model.onnx", None)
test_input_X = numpy.random.randn(4, 5).astype(numpy.float32)
test_input_Y = numpy.random.randn(4, 5).astype(numpy.float32)
# gets X in inputs[0] and Y in inputs[1]
inputs = session.get_inputs()
# gets Z in outputs[0]
outputs = session.get_outputs()
test_input = numpy.array([test_input_X, test_input_Y])

test_output = numpy.array(
    [
        session.run(
            [outputs[0].name],
            {inputs[0].name: test_input_X, inputs[1].name: test_input_Y},
        )[0]
    ],
    dtype=numpy.float32,
)
print("Input:", test_input)
print("Output:", test_output)
