# Tensorflow implementation of model with converter

You can use this script to converter origin model to tensorflow and tflite version.

## Run
Covert model to Tensorflow
```
 python3 TFConverter.py --pytorch_model pytorchWeightPath.pt --postprocess true/false
```

Export model to TFlite with full int8 Post-Training Quantization (optinal)
```
 python3 exportTFlitePostQuant.py --save_dir tenforflowWeightPath.keras 
                                  --out TFlightWeightDir 
                                  --quantize_int8 true/false 
                                  --repr_data representDataDir
```
