import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

df1 = pd.read_csv("data/ionosphere.data")

#Data cleanup
def cleanData(dataset):
	#1. Eliminate duplicates
	dataset = dataset.drop_duplicates(subset=None, keep = 'first', inplace = False)
	#2. Check categorical variables for consistency
	#3. One hot encoding
	# dummy = pd.get_dummies(dataset.Output)
	# dummy.head()
	# dataset = pd.concat([dataset, dummy], axis=1)
	# dataset.head()

	#4. Eliminate missing values
	dataset = dataset.dropna()

	# Drop columns where all the values are the same as they offer us no information
	cols = dataset.select_dtypes([np.number]).columns
	std = dataset[cols].std()
	cols_to_drop = std[std==0].index
	dataset = dataset.drop(cols_to_drop, axis=1)
	# dataset.insert(0,{'Att1'})
	return dataset

class naive_bayes:
	X = []
	y = []
	likelihood_type = []
	predictor_count = 0
	prior_zero = 0
	prior_one = 0
	instances = 0
	zero_mean_vals = []
	zero_std_vals = []
	one_mean_vals = []
	one_std_vals = []
	output_values = []

	#Compute the prior values of each output
	def compute_priors(self, data ):
		for i in range(len(data)):
			if data[i] == self.output_values[0]:
				self.prior_zero = self.prior_zero + 1
			if data[i] == self.output_values[1]:
				self.prior_one = self.prior_one + 1
		self.prior_zero = self.prior_zero/len(data)
		self.prior_one = self.prior_one/len(data)

	def init(self, dataset, likelihood_type, output_values, instances, predictor_count):
		#Initialize some variables
		self.predictor_count = predictor_count
		self.output_values = output_values
		self.instances = instances

		# X = feature values, all the columns except the last column
		self.X = dataset.iloc[:, :-1]

    	# y = target values, last column of the data frame
		self.y = dataset.iloc[:, -1]
		self.X = np.c_[np.ones((self.X.shape[0], 1)), self.X] #Transform input into array
		self.y = self.y[:, np.newaxis] #Transform y into 2D array
		self.X = np.delete(self.X, 0, 1) #Remove the first column consiting of nothing but 1s

    	# For every predictor, a different type of likelihood. 0 is bernoulli, 1 is gaussian
    	# and 2 is multinomial
		for i in range(len(likelihood_type)):
			self.likelihood_type.append(likelihood_type[i])

	#Get the mean and standard deviation of a set of data for each output
	def calc_mean_and_std(self, data):

		zero_data = []
		one_data = []
		#For each instance, check if the output is 0 or 1 and separate the data
		#Appropriately
		for i in range(len(data)):
			if self.y[i][0]==self.output_values[0]:
				zero_data.append(data[i])
			elif self.y[i][0] == self.output_values[1]:
				one_data.append(data[i])
		zero_mu = np.mean(zero_data)
		zero_std = np.std(zero_data)
		one_mu = np.mean(one_data)
		one_std = np.std(one_data)
		return(zero_mu,zero_std,one_mu, one_std)

	#Get the gaussian likelihood  of an attribute for 0 and 1 outputs
	def gaussian_likelihood(self, x, mean, std):
		var = float(std) ** 2
		denominator = (2*math.pi*var)**.5
		numerator = math.exp(-(float(x)-float(mean))**2/(2*var))
		return numerator/denominator

	def fit(self):
		self.compute_priors(self, self.y)
		# Set up all the means and std devs
		for column in self.X.T:
			# print(column)
			res = self.calc_mean_and_std(self, column)
			# print('test2')
			self.zero_mean_vals.append(res[0])
			self.zero_std_vals.append(res[1])
			self.one_mean_vals.append(res[2])
			self.one_std_vals.append(res[3])

	#Compute the evidence
	#UNUSUED for now as it creates problems and doesn't impact the outcome
	def calculate_evidence(self, params):
		result = 1;
		for i in range(len(params)):
			count=0
			for row in self.X.T:
				if row[i] == params[i]:
					count = count + 1
			result = result * count/self.instances			
		return result

	# prediction function
	def predict(self, data_point):
		evidence = self.calculate_evidence(self, data_point)
		#Initialize some variables
		zero_tmp_likelihood = []
		one_tmp_likelihood = []
		zero_total_likelihood=1
		one_total_likelihood=1
		for i in range(self.predictor_count):
			
			
			if self.likelihood_type[i] == 1: #Likelihood type of 1 means gaussian likelihood
				# Workaround for when the standard deviation of a feature is 0 for one of the outputs
				if self.zero_std_vals[i] == 0:
					zero_tmp_likelihood.append(1)
				else:
					zero_tmp_likelihood.append(self.gaussian_likelihood(self, data_point[i], self.zero_mean_vals[i], self.zero_std_vals[i]))
				if self.one_std_vals[i] == 0:
					one_tmp_likelihood.append(1)
				else:
					one_tmp_likelihood.append(self.gaussian_likelihood(self, data_point[i], self.one_mean_vals[i], self.one_std_vals[i]))
			#Compute the likelihood of our datapoint for both 0 and 1 outputs				
			zero_total_likelihood = zero_total_likelihood * zero_tmp_likelihood[i]
			one_total_likelihood = one_total_likelihood * one_tmp_likelihood[i]
		#Compute probably of 0 and 1 outputs using our likelihood
		p_zero = self.prior_zero * zero_total_likelihood
		p_one = self.prior_one * one_total_likelihood 
		#Output the output with higher probability
		if p_zero>p_one:
			return self.output_values[0]
		else: 
			return self.output_values[1]

if __name__ == "__main__":
	df1 = cleanData(df1)
	x = naive_bayes
	likelihood = [None] * (df1.shape[1]-1)

	# -------------------------INONOSPHERE SETUP------------------------
	# For Ionosphere, all 34 data features are continuous.
	for i in range(0,(df1.shape[1]-1)):
		likelihood[i] = 1
	out = df1.Output.unique()
	output_values = [None] * 2
	#Initialize our model with our data
	x.init(x, df1, likelihood, out, df1.shape[0], df1.shape[1]-1)
	x.fit(x)
	#Note: remove the second predictor from the input array below since the second column
	# of the dataframe has values of all 0s, it is excluded from the model.
	test = [1,0,0.36876,-1,-1,-1,-0.07661,1,1,0.95041,0.74597,-0.38710,-1,-0.79313,-0.09677,1,0.48684,0.46502,0.31755,-0.27461,-0.14343,-0.20188,-0.11976,0.06895,0.03021,0.06639,0.03443,-0.01186,-0.00403,-0.01672,-0.00761,0.00108,0.00015,0.00325]
	#Since the second predictor is always the same for all instances, remove it from the input
	test.pop(1)
	res = x.predict(x, test)
	print("Predicted output: ", res)
	# -------------------------/IONOSPHERE SETUP--------------------------


