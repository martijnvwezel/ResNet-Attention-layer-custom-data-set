# ResNet with attention-layer on custom data set.
Here is a ResNet with attention layers that is designed for custom data sets. There is always an way for improvements, but this would get you started. `The training.py` is compatible with the cifar data sets. The attention layer is based on the following [github page](https://github.com/qubvel/residual_attention_network) (commit version: 15c111d). 

# Usage
Install with anaconda python 3 version and keras. Go to the train directory.   
Change in `vars.py` the variables for your data set.  
  
 **Always double check if function `load_custom_data(...)` is uncomment if your learn on your own data set**
``` bash 
# Start training
python training.py

```

# Data set structure
The directory names (train, validation, test) can be changed in the `vars.py` file if needed.   
In the `vars.py` there are some defines that depense on your data set, like the class_names.
```
Data set structure: 
        [path_to_dataset]/train/[class_directorys]/[files]
        [path_to_dataset]/validation/[class_directorys]/[files]
        [path_to_dataset]/test/[class_directorys]/[files]

```

# Made by:  
martijnvwezel@muino.nl and rens@rens.nu