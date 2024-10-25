# Evaluation
Python Evaluation Code for Test Dataset


## Usage

##### Run evaluationTflite.py or evaluationPytorch (coming soon)

````
python3 evaluationTflite.py
````
##### Before calculating metrics 

````
python3 setup.py build_ext --inplace
````

##### Calculating mAP@50 and mAP@50-95
````
python3 evaluation.py -p <your prediction dir> -g <groud truth dir>
````
