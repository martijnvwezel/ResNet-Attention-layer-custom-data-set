# ResNet with attention-layer on custom data set.
Here is a ResNet with attention layers that are designed for custom data sets. There is always a way for improvements, but this would get you started. `The training.py` is compatible with the CIFAR data sets. The attention layer is based on the following [github page](https://github.com/qubvel/residual_attention_network) (commit version: 15c111d).   

*This is the cleaned version, maybe some mistakes namings are wrong, like the test script is missing*
# Usage
Install with anaconda python 3 version and Keras. Go to the train directory.   
Change in `vars.py` the variables for your dataset.  
  
 **Always double check if function `load_custom_data(...)` is uncommented if you learn on your own data set**
``` bash 
# Start training
python training.py

```

# Dataset structure
The directory names (train, validation, test) can be changed in the `vars.py` file if needed.   
In the `vars.py` there are some defines that depense on your data set, like the class_names.
```
Dataset structure: 
        [path_to_dataset]/train/[class_directorys]/[files]
        [path_to_dataset]/validation/[class_directorys]/[files]
        [path_to_dataset]/test/[class_directorys]/[files]

```

# Made by:  
martijnvwezel@muino.nl and [@RensHam](https://github.com/RensHam)
