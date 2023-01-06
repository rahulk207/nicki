##NICKI-NICKI(hide)

#parts of the code are from https://github.com/tkipf/pygcn

the code is set for training NICKI for cora dataset with base class 5 and target class 0

for running NICKI(hide) -- 
	uncomment first line in script.sh
        put value: HIDDEN=True in main_new.py

for changing base class --
	change value of base_class variable in main_new.py and feature_generator_vae.py

for changing target class --
	change value 359 in line: num_fake_nodes = int(budget_coeff*359) to number of target nodes in test set, in main_new.py
	change num_of_target_nodes = 359 value also in feature_generator_vae.py
	change parameter -target to the designated target class in script.sh 'currently for class 0: -target 0'

for budget change --
	update first line of script.sh with required budget, currently 0.03

for running other datasets --
	datasets are available in dataset folder, refer to paper details for changes in code.


run the code using command --
	sh script.sh
