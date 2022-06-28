import tensorrt as trt 
import argparse 
import os.path as osp

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_model_name",type=str)
    args = parser.parse_args()
    return args

def tensorrt_serialize(onnx_model_name,output_name):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network,TRT_LOGGER)
    with open(onnx_model_name, 'rb') as model:
        parser.parse(model.read())
    print("print parser error:")
    for index in range(parser.num_errors):
        print(parser.get_error(index))
    # output_tensors = [network.get_output(i) for i in range(network.num_outputs)]
    max_batch_size=1
    builder.max_batch_size = max_batch_size

    builder.max_workspace_size = 1 <<  20 # This determines the amount of memory available to the builder when building an optimized engine and should generally be set as high as possible.
    # with trt.Builder(TRT_LOGGER) as builder:
    engine = builder.build_cuda_engine(network)

    with open(output_name, "wb") as f:
            f.write(engine.serialize())
    # with open(output_name, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    #         engine = runtime.deserialize_cuda_engine(f.read())
    print("{} to tensorrt model has been successfully transformed".format(osp.splitext(osp.basename(output_name))[0]))

if __name__=="__main__":
    args = parse_args()
    onnx_model_name = args.onnx_model_name
    output_name = onnx_model_name.replace(".onnx",".engine")
    tensorrt_serialize(onnx_model_name,output_name)