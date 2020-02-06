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
	# print(dataset.Output.value_counts())
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
		self.predictor_count = predictor_count
		self.output_values = output_values
		self.instances = instances
		# X = feature values, all the columns except the last column
		self.X = dataset.iloc[:, :-1]
		# print(self.X)

    	# y = target values, last column of the data frame
		self.y = dataset.iloc[:, -1]
		self.X = np.c_[np.ones((self.X.shape[0], 1)), self.X] #Transform input into array
		self.y = self.y[:, np.newaxis] #Transform y into 2D array
		self.X = np.delete(self.X, 0, 1) #Remove the first column consiting of nothing but 1s
		# print("Self.X: ")
		# print(self.X)

    	# For every predictor, a different type of likelihood. 0 is bernoulli, 1 is gaussian
    	# and 2 is multinomial
		for i in range(len(likelihood_type)):
			self.likelihood_type.append(likelihood_type[i])

	#Get the mean and standard deviation of a set of data
	def calc_mean_and_std(self, data):
		# print(data)
		zero_data = []
		one_data = []
		for i in range(len(data)):
			if self.y[i][0]==output_values[0]:
				zero_data.append(data[i])
			elif self.y[i][0] == output_values[1]:
				one_data.append(data[i])
		zero_mu = np.mean(zero_data)
		zero_std = np.std(zero_data)
		one_mu = np.mean(one_data)
		one_std = np.std(one_data)
		return(zero_mu,zero_std,one_mu, one_std)

	#Get the gaussian likelihood  of an attribute
	def gaussian_likelihood(self, x, mean, std):
		# print("x: ", x)
		# print("std: ", std)
		var = float(std) ** 2
		denominator = (2*math.pi*var)**.5
		numerator = math.exp(-(float(x)-float(mean))**2/(2*var))
		return numerator/denominator

	def fit(self):
		self.compute_priors(self, self.y)
		# Set up all the means and std devs
		# print(self.X.T) 
		for column in self.X.T:
			res = self.calc_mean_and_std(self, column)
			self.zero_mean_vals.append(res[0])
			self.zero_std_vals.append(res[1])
			self.one_mean_vals.append(res[2])
			self.one_std_vals.append(res[3])

	#Compute the evidence
	def calculate_evidence(self, params):
		result = 1;
		for i in range(len(params)):
			count=0
			for row in self.X.T:
				if row[i] == params[i]:
					count = count + 1
			# print("Param: ", params[i])
			# print("Count: ", count)
			result = result * count/self.instances			
		return result

	# prediction function
	def predict(self, data_point):
		# p_zero = 0
		# p_one = 0
		evidence = self.calculate_evidence(self, data_point)
		zero_tmp_likelihood = []
		one_tmp_likelihood = []
		zero_total_likelihood=1
		one_total_likelihood=1
		# print(self.predictor_count)
		for i in range(self.predictor_count):
			# print(i)
			if self.likelihood_type[i] == 1:
				#Workaround for when the standard deviation of a feature is 0 for a certain output
				if self.zero_std_vals[i] == 0:
					self.zero_std_vals[i] = self.one_std_vals[i]
				if self.one_std_vals[i] == 0:
					self.one_std_vals[i] = self.zero_std_vals[i]
			#Compute the likelihood of our datapoint for both 0 and 1 outputs
				zero_tmp_likelihood.append(self.gaussian_likelihood(self, data_point[i], self.zero_mean_vals[i], self.zero_std_vals[i]))
				one_tmp_likelihood.append(self.gaussian_likelihood(self, data_point[i], self.one_mean_vals[i], self.one_std_vals[i]))
			zero_total_likelihood = zero_total_likelihood * zero_tmp_likelihood[i]
			one_total_likelihood = one_total_likelihood * one_tmp_likelihood[i]
		p_zero = self.prior_zero * zero_total_likelihood
		p_one = self.prior_one * one_total_likelihood 
		# print("P_ZERO: ", p_zero)
		# print("P_ONE: ", p_one)
		if p_zero>p_one:
			return output_values[0]
		else: 
			return output_values[1]




    # def fit(self):



if __name__ == "__main__":
	df1 = cleanData(df1)
	x = naive_bayes
	likelihood = [None] * (df1.shape[1]-1)

	# print(df1)
	# For Ionosphere, all 34 data features are continuous.
	for i in range(0,(df1.shape[1]-1)):
		likelihood[i] = 1
	out = df1.Output.unique()
	output_values = [None] * 2
	output_values[0] = out[0]
	output_values[1] = out[1]
	# print(output_values)
	#Initialize our model with our data
	x.init(x, df1, likelihood, output_values, df1.shape[0], df1.shape[1]-1)
	x.fit(x)

	#Note: remove the second predictor from the input array below since the second column
	# of the dataframe has values of all 0s, it is excluded from the model.
	test = [1,0.58940,-0.60927,0.85430,0.55298,0.81126,0.07285,0.56623,0.16225,0.32781,0.24172,0.50331,0.12252,0.63907,0.19868,0.71854,0.42715,0.54305,0.13907,0.65232,0.27815,0.68874,0.07285,0.51872,0.26653,0.49013,0.27687,0.46216,0.28574,0.43484,0.29324,0.40821,0.29942]
	res = x.predict(x, test)
	print("Predicted output: ", res)


