import tensorflow as tf
import numpy as np

m = 'dms_tflite_models/yolov8n_smk_call.tflite'
interp = tf.lite.Interpreter(model_path=m)
interp.allocate_tensors()
inp = interp.get_input_details()
outs = interp.get_output_details()

print('=== INPUT ===')
for d in inp:
    print(f"  shape={d['shape']}  dtype={d['dtype']}  quant={d['quantization']}")

print('=== OUTPUTS ===')
for i, d in enumerate(outs):
    print(f"  [{i}] name={d['name']}  shape={d['shape']}  dtype={d['dtype']}  quant={d['quantization']}")

print(f"  Total outputs: {len(outs)}")

# Now run with a real-looking input (random noise)
in_shape = inp[0]['shape']
dummy = (np.random.rand(*in_shape).astype(np.float32))
interp.set_tensor(inp[0]['index'], dummy)
interp.invoke()

print('\n=== OUTPUT VALUES (random input) ===')
for i, d in enumerate(outs):
    raw = interp.get_tensor(d['index'])
    print(f"  [{i}] shape={raw.shape}  min={raw.min():.5f}  max={raw.max():.5f}  mean={raw.mean():.5f}")
    # Show a sample of values
    flat = raw.flatten()
    print(f"       first 10 values: {flat[:10].tolist()}")
