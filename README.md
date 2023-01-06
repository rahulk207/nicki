## NICKI-NICKI(hide)

Parts of the code are from https://github.com/tkipf/pygcn. <br />
Our datasets can be found [here](https://drive.google.com/file/d/1W9oW_vANjO6i39q8ZtLg7rXlqmkIN5dQ/view?usp=sharing).

The code is set for training NICKI for cora dataset with base class 5 and target class 0. <br />

### Steps to run the pipeline
* For running NICKI(hide) -
	* Uncomment first line in script.sh
	* Put value: HIDDEN=True in main_new.py

* For changing base class-
	* Change value of base_class variable in main_new.py and feature_generator_vae.py

* For changing target class-
	* Change value 359 in line: num_fake_nodes = int(budget_coeff*359) to number of target nodes in test set, in main_new.py
	* Change num_of_target_nodes = 359 value also in feature_generator_vae.py
	* Change parameter -target to the designated target class in script.sh 'currently for class 0: -target 0'

* For budget change-
	* Update first line of script.sh with required budget, currently 0.03

* For running other datasets-
	* Datasets are available in dataset folder, refer to paper details for changes in code.


* Run the code using command - `sh script.sh`
