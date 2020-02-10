import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import sys

filename=sys.argv[1]
df1 = pd.read_csv(filename)

#Data cleanup
def cleanData(dataset):
	#1. Eliminate duplicates
	dataset = dataset.drop_duplicates(subset=None, keep = 'first', inplace = False)

	#4. Eliminate missing values (ionosphere has no missing values)
	dataset = dataset.dropna()
	dataset = dataset[~dataset['Att1'].isin(['?'])]
	dataset = dataset[~dataset['Att2'].isin(['?'])]
	dataset = dataset[~dataset['Att3'].isin(['?'])]
	dataset = dataset[~dataset['Att4'].isin(['?'])]
	dataset = dataset[~dataset['Att5'].isin(['?'])]
	dataset = dataset[~dataset['Att6'].isin(['?'])]
	dataset = dataset[~dataset['Att7'].isin(['?'])]
	dataset = dataset[~dataset['Att8'].isin(['?'])]
	dataset = dataset[~dataset['Att9'].isin(['?'])]
	if "adult" in filename or "hepatitis" in filename:
		dataset = dataset[~dataset['Att10'].isin(['?'])]
		dataset = dataset[~dataset['Att11'].isin(['?'])]
		dataset = dataset[~dataset['Att12'].isin(['?'])]
		dataset = dataset[~dataset['Att13'].isin(['?'])]
		dataset = dataset[~dataset['Att14'].isin(['?'])]
	if "hepatitis" in filename:
		dataset = dataset[~dataset['Att15'].isin(['?'])]
		dataset = dataset[~dataset['Att16'].isin(['?'])]
		dataset = dataset[~dataset['Att17'].isin(['?'])]
		dataset = dataset[~dataset['Att18'].isin(['?'])]
		dataset = dataset[~dataset['Att19'].isin(['?'])]
	# Drop columns where all the values are the same as they offer us no information
	cols = dataset.select_dtypes([np.number]).columns
	std = dataset[cols].std()
	cols_to_drop = std[std==0].index
	dataset = dataset.drop(cols_to_drop, axis=1)
	# dataset.insert(0,{'Att1'})
	return dataset

class naive_bayes:
	X = [] #List of predictors
	y = [] #List of outputs
	likelihood_type = [] #List containing information as to what type of likelihood each predictor should use
	predictor_count = 0 #Total amount of predictors
	prior_zero = 0 #Prior of the 0-output
	prior_one = 0 #Prior of the 1-output
	instances = 0 #Number of instances in the dataset
	zero_data = [] #List holding all the data with 0-output
	one_data = [] #List holding all the data with 1-output
	zero_count_vals = [] #List of counts of values of a predictor for 0-outputs
	zero_mean_vals = [] #List of mean values of predictors for 0-outputs
	zero_std_vals = [] #List of standard deviation values for predictors for 0-outputs
	one_count_vals = [] #List of counts of values of a predictor for 1-outputs
	one_mean_vals = [] #List of mean values of predictors for 1-outputs
	one_std_vals = [] #List of standard deviation values for predictors for 1-outputs
	output_values = [] #Literal binary output values (e.g. "b" and "g" for ionosphere)

	#Compute the prior values of each output
	def compute_priors(self, data):
		for i in range(len(data)):
			if data[i] == self.output_values[0]:
				self.prior_zero = self.prior_zero + 1
			if data[i] == self.output_values[1]:
				self.prior_one = self.prior_one + 1
		self.prior_zero = self.prior_zero/len(data)
		self.prior_one = self.prior_one/len(data)

	def init(self, dataset, likelihood_type, output_values, instances, predictor_count, y_pos):
		#Initialize some variables
		self.predictor_count = predictor_count
		self.output_values = output_values
		self.instances = instances
		if y_pos == "last":
    		# y = target values, last column of the data frame
			self.y = dataset.iloc[:, -1]
			# X = feature values, all the columns except the last column
			self.X = dataset.iloc[:, :-1]
		elif y_pos == "first":
			# y = target values, first column of the data frame
			self.y=dataset.iloc[:, 0]
			# X = feature values, all the columns except the first column
			self.X = dataset.iloc[:, 1:]
		self.y = self.y[:, np.newaxis] #Transform y into 2D array
		self.X = np.c_[np.ones((self.X.shape[0], 1)), self.X] #Transform input into array
		self.X = np.delete(self.X, 0, 1) #Remove the first column consiting of nothing but 1s

    	# For every predictor, a different type of likelihood. 0 is bernoulli, 1 is gaussian
    	# and 2 is multinomial
		for i in range(len(likelihood_type)):
			self.likelihood_type.append(likelihood_type[i])

	#Get the mean and standard deviation of a set of data for each output
	def calc_mean_and_std(self, data, y_data , behaviour):
		zero_data = []
		one_data = []
		if behaviour == 0:
		#For each instance, check if the output is 0 or 1 and separate the data
		#Appropriately
			for i in range(len(data)):
				if self.y[i][0]==self.output_values[0]:
					if "hepatitis" in filename:
						data[i] = float(data[i])
					zero_data.append(data[i])
				elif self.y[i][0] == self.output_values[1]:
					if "hepatitis" in filename:
						data[i] = float(data[i])
					one_data.append(data[i])
			if isinstance(zero_data[0], int) or isinstance(zero_data[0], float):
				zero_mu = np.mean(zero_data) #Mean value of predictor for 0-outputs
				zero_std = np.std(zero_data) #Standard deviation value of predictor for 0-outputs
				one_mu = np.mean(one_data) #Mean value of predictor for 1-outputs
				one_std = np.std(one_data) #Standard deviation value of predictor for 1-outputs
			else:
				zero_mu = None
				zero_std = None
				one_mu = None
				one_std = None
			zero_counts = len(zero_data) #Count of amount of times predictor's value appears for 0-outputs
			one_counts = len(one_data) #Count of amount of times predictor's value appears for 1-outputs
		elif behaviour==1:
			# print(len(data))
			for i in range(len(data)):
				if y_data[i]==self.output_values[0]:
					if "hepatitis" in filename:
						data[i] = float(data[i])
					zero_data.append(data[i])
				elif y_data[i] == self.output_values[1]:
					if "hepatitis" in filename:
						data[i] = float(data[i])
					one_data.append(data[i])
			if isinstance(zero_data[0], int) or isinstance(zero_data[0], float):
				zero_mu = np.mean(zero_data) #Mean value of predictor for 0-outputs
				zero_std = np.std(zero_data) #Standard deviation value of predictor for 0-outputs
				one_mu = np.mean(one_data) #Mean value of predictor for 1-outputs
				one_std = np.std(one_data) #Standard deviation value of predictor for 1-outputs
			else:
				zero_mu = None
				zero_std = None
				one_mu = None
				one_std = None
			zero_counts = len(zero_data) #Count of amount of times predictor's value appears for 0-outputs
			one_counts = len(one_data) #Count of amount of times predictor's value appears for 1-outputs
		return(zero_mu,zero_std,one_mu, one_std, zero_counts, one_counts)

	#Get the gaussian likelihood  of an attribute for 0 and 1 outputs
	def gaussian_likelihood(self, x, mean, std):
		var = float(std) ** 2
		denominator = (2*math.pi*var)**.5
		numerator = math.exp(-(float(x)-float(mean))**2/(2*var))
		return numerator/denominator

	#Get the bernoulli likelihood
	def bernoulli_likelihood(self, output_type, feature_index, obs):
		total = 0
		if output_type == 0:
			total = len(self.zero_data)
			count = 0
			for row in self.zero_data:
				tmp=0
				if isinstance(row[feature_index], int) or isinstance(row[feature_index], float):
					tmp = row[feature_index]
				else:
					tmp = row[feature_index].strip()
				if tmp == obs:
					count = count + 1
		elif output_type == 1:
			total = len(self.one_data)
			count = 0
			for row in self.one_data:
				tmp=0
				if isinstance(row[feature_index], int) or isinstance(row[feature_index], float):
					tmp = row[feature_index]
				else:
					tmp = row[feature_index].strip()
				if tmp == obs:
					count = count + 1
		if total == 0:
			return 0 
		else:
			lh = count/total
		return lh

	#Get the multinomial likelihood
	def multivar_likelihood(self, output_type, feature_index, obs):
		total = 0
		if output_type == 0:
			total = len(self.zero_data)
			count = 0
			for row in self.zero_data:
				# print("----------")
				# print(row[feature_index])
				# print(obs)
				tmp = 0
				if isinstance(row[feature_index], int) or isinstance(row[feature_index], float):
					tmp = row[feature_index]
				else:
					tmp = row[feature_index].strip()
				if tmp == obs:
					count = count + 1
		elif output_type == 1:
			total = len(self.one_data)
			count = 0
			for row in self.one_data:
				tmp=0
				if isinstance(row[feature_index], int) or isinstance(row[feature_index], float):
					tmp = row[feature_index]
				else:
					tmp = row[feature_index].strip()
				if tmp == obs:
					count = count + 1
		if(total == 0):
			return 0
		else:
			lh = count/total
		return lh

	def fit(self):
		self.compute_priors(self, self.y)
		#For each instance, check if the output is 0 or 1 and separate the data
		#Appropriately
		for i in range(self.instances):
			if self.y[i][0]==self.output_values[0]:
				self.zero_data.append(self.X[i])
			elif self.y[i][0] == self.output_values[1]:
				self.one_data.append(self.X[i])
		# Set up all the means and std devs
		for column in self.X.T:
			res = self.calc_mean_and_std(self, column, [], 0)
			self.zero_mean_vals.append(res[0])
			self.zero_std_vals.append(res[1])
			self.one_mean_vals.append(res[2])
			self.one_std_vals.append(res[3])
			self.zero_count_vals.append(res[4])
			self.one_count_vals.append(res[5])

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
	def predict(self, data_points):
		# evidence = self.calculate_evidence(self, data_point)
		#Initialize some variables
		results = []
		counter = 0
		for point in data_points:
			counter = counter + 1
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
						zero_tmp_likelihood.append(self.gaussian_likelihood(self, point[i], self.zero_mean_vals[i], self.zero_std_vals[i]))
					if self.one_std_vals[i] == 0:
						one_tmp_likelihood.append(1)
					else:
						one_tmp_likelihood.append(self.gaussian_likelihood(self, point[i], self.one_mean_vals[i], self.one_std_vals[i]))
				elif self.likelihood_type[i] == 0:
					zero_tmp_likelihood.append(self.bernoulli_likelihood(self, 0, i, point[i]))
					one_tmp_likelihood.append(self.bernoulli_likelihood(self, 1, i, point[i]))
				elif self.likelihood_type[i] == 2:
					zero_tmp_likelihood.append(self.multivar_likelihood(self, 0, i, point[i]))
					one_tmp_likelihood.append(self.multivar_likelihood(self, 1, i, point[i]))

			#Compute the likelihood of our datapoint for both 0 and 1 outputs
				# print(zero_tmp_likelihood)			
				zero_total_likelihood = zero_total_likelihood * zero_tmp_likelihood[i]
				one_total_likelihood = one_total_likelihood * one_tmp_likelihood[i]
		#Compute probably of 0 and 1 outputs using our likelihood
			p_zero = self.prior_zero * zero_total_likelihood
			p_one = self.prior_one * one_total_likelihood 
		#Output the output with higher probability
			if p_zero>p_one:
				results.append(self.output_values[0])
			else: 
				results.append(self.output_values[1])
		return results
	#k-fold cross validation
	def accuracy(self, k):
		counter = 0
		tmp_X = np.c_[np.ones((self.X.shape[0], 1)), self.X]
		expected_results = []
		for row in tmp_X:
			row[0] = counter
			counter = counter + 1
		np.random.shuffle(tmp_X)
		
		for row in tmp_X:
			expected_results.append(self.y[int(row[0])][0])
		tmp_X = np.delete(tmp_X, 0, 1)
		fold_length = int(self.instances/k)
		data_folds = [[[]]]
		for i in range(k):
			j=0
			while j<fold_length:
				for l in range(self.predictor_count):
					data_folds[i][j].append(tmp_X[j+i*fold_length][l])
					# print(data_folds)
				j = j + 1
					# print("FUCK YOU")
				if j<fold_length: 	
					data_folds[i].append([])
			data_folds.append([[]])
		#For each instance, check if the output is 0 or 1 and separate the data
		#Appropriately
		results = []
		for i in range(k):
			#Re-initialize the variables
			self.zero_data = []
			self.one_data = []
			self.zero_count_vals = [] 
			self.zero_mean_vals = []
			self.zero_std_vals = [] 
			self.one_count_vals = []
			self.one_mean_vals = []
			self.one_std_vals = []
			train_data = []
			test_data = []
			train_y = []
			test_y = []
			for m in range(k):
				# print("m: ", m)
				if i == m:
					test_data = data_folds[i]
					for l in range(fold_length):
						test_y.append(expected_results[i*fold_length+m])
				else:
					for l in range(fold_length):
						train_data.append(data_folds[i][l])
						if expected_results[i*fold_length+m]==self.output_values[0]:
							self.zero_data.append(data_folds[i][l])
						elif expected_results[i*fold_length+m] == self.output_values[1]:
							self.one_data.append(data_folds[i][l])
						train_y.append(expected_results[i*fold_length+m])
			self.compute_priors(self, train_y)
			# print(train_data)
		# Set up all the means and std devs
			for column in list(map(list, zip(*train_data))):
				res = self.calc_mean_and_std(self, column, train_y, 1)
				self.zero_mean_vals.append(res[0])
				self.zero_std_vals.append(res[1])
				self.one_mean_vals.append(res[2])
				self.one_std_vals.append(res[3])
				self.zero_count_vals.append(res[4])
				self.one_count_vals.append(res[5])

			res = self.predict(self, test_data)
			for a in res:
				results.append(a)
		success = 0
		for i in range(len(results)):	
			if results[i] == expected_results[i]:
				success = success + 1
		acc = success/len(results)
		print("Accuracy rate: ", acc*100, "%")
		return acc


if __name__ == "__main__":

	df1 = cleanData(df1)
	x = naive_bayes
	likelihood = [None] * (df1.shape[1]-1)

	# -------------------------INONOSPHERE SETUP------------------------
	if "ionosphere" in filename:
		#The last column is the output column
		ypos = "last"
		# For Ionosphere, all 34 data features are continuous.
		for i in range(0,(df1.shape[1]-1)):
			likelihood[i] = 1
		out = df1.Output.unique()
		#Initialize our model with our data
		x.init(x,df1,likelihood,out,df1.shape[0],df1.shape[1]-1, ypos)
		x.accuracy(x,5)
		# x.fit(x)
		test_X = [[1,0,0.36876,-1,-1,-1,-0.07661,1,1,0.95041,0.74597,-0.38710,-1,-0.79313,-0.09677,1,0.48684,0.46502,0.31755,-0.27461,-0.14343,-0.20188,-0.11976,0.06895,0.03021,0.06639,0.03443,-0.01186,-0.00403,-0.01672,-0.00761,0.00108,0.00015,0.00325], [1,0,1,-0.08183,1,-0.11326,0.99246,-0.29802,1,-0.33075,0.96662,-0.34281,0.85788,-0.47265,0.91904,-0.48170,0.73084,-0.65224,0.68131,-0.63544,0.82450,-0.78316,0.58829,-0.74785,0.67033,-0.96296,0.48757,-0.85669,0.37941,-0.83893,0.24117,-0.88846,0.29221,-0.89621], [1,0,0.01975,0.00705,0.04090,-0.00846,0.02116,0.01128,0.01128,0.04372,0.00282,0.00141,0.01975,-0.03103,-0.01975,0.06065,-0.04090,0.02680,-0.02398,-0.00423,0.04372,-0.02539,0.01834,0,0,-0.01269,0.01834,-0.01128,0.00564,-0.01551,-0.01693,-0.02398,0.00705,0]]
		#Since the second predictor is always the same for all instances, remove it from each input point
		# for item in test_X:
		# 	item.pop(1)
		# # res = x.predict(x, test_X)
		# # print("Predicted outputs: ", res)
		# # acc = x.accuracy(x,5)
		# print("Ionosphere data model accuracy: ", acc*100, "%")
	# -------------------------/IONOSPHERE SETUP--------------------------

	# -------------------------ADULT SETUP--------------------------------
	elif "adult" in filename:
		#The last column is the output column
		ypos="last"
		likelihood[0]=1 #Age; continuous
		likelihood[1]=2 #Workclass; categorical
		likelihood[2]=1 #fnlwgt; continuous
		likelihood[3]=2 #education; categorical
		likelihood[4]=1 #education-num; continuous
		likelihood[5]=2 #marital-status; categorical
		likelihood[6]=2 #occupation; categorical
		likelihood[7]=2 #relationship; categorical
		likelihood[8]=2 #race; categorical
		likelihood[9]=0 #sex; bernoulli
		likelihood[10]=1 #capital gain; continuous
		likelihood[11]=1 #capital loss; continuous
		likelihood[12]=1 #hours per week; continuous
		likelihood[13]=2 #native country; categorical 
		out = df1.Output.unique()
		#Initialize our model with our data
		x.init(x,df1,likelihood,out,df1.shape[0],df1.shape[1]-1, ypos)
		# x.fit(x)
		x.accuracy(x,5)
		input_vector = [[30, "State-gov", 141297, "Bachelors", 13, "Married-civ-spouse", "Prof-specialty", "Husband", "Asian-Pac-Islander", "Male", 0, 0, 40, "India"],[39, "Private", 367260, "HS-grad", 9, "Divorced", "Exec-managerial", "Not-in-family", "White", "Male", 0, 0, 80, "United-States"]]
		# res = x.predict(x, input_vector)
		# print("Predicted outputs: ", res)
		# print("Adult data model accuracy: ", acc*100, "%")
	# ------------------------/ADULT SETUP---------------------------------

	# ------------------------BREAST CANCER SETUP--------------------------
	elif "breast-cancer" in filename:
		#The first column is the output column
		ypos = "first"
		likelihood[0]=2 #Age; categorical
		likelihood[1]=2 #Menopause; categorical
		likelihood[2]=2 #tumor size; categorical 
		likelihood[3]=2 #inv-nodes; categorical
		likelihood[4]=0 #node caps; bernoulli
		likelihood[5]=2 #deg malig; categorical
		likelihood[6]=0 #breast; bernoulli
		likelihood[7]=2 #breast-quad; categorical
		likelihood[8]=0 #irradiat; bernoulli
		out=df1.Output.unique()
		# #Initialize our model with our data
		x.init(x,df1,likelihood,out,df1.shape[0],df1.shape[1]-1, ypos)
		x.accuracy(x,5)
		# x.fit(x)
		input_vector =[["60-69","ge40","15-19","0-2","no",2,"left","left_low","no"],["40-49","premeno","30-34","0-2","yes",3,"right","right_up","no"]]
		# res = x.predict(x, input_vector)
		# print("Predicted outputs: ", res)
		# acc = x.accuracy(x,5)
		# print("Breast-cancer data model accuracy: ", acc*100, "%")
	# ------------------------/BREAST CANCER SETUP-------------------------

	# ------------------------HEPATITIS SETUP------------------------------

	elif "hepatitis" in filename:
		ypos="first"
		likelihood[0]=1 #Age; Continuous
		likelihood[1]=0 #Sex; bernoulli
		likelihood[2]=0 #Steroid; bernoulli
		likelihood[3]=0 #Antivirals; bernoulli
		likelihood[4]=0 #Fatigue; bernoulli
		likelihood[5]=0 #Malaise; bernoulli
		likelihood[6]=0 #Anoerixa; bernoulli
		likelihood[7]=0 #Liver big; bernoulli
		likelihood[8]=0 #Liver firm; bernoulli
		likelihood[9]=0 #Spleen palpable; bernoulli
		likelihood[10]=0 #Spiders; bernoulli
		likelihood[11]=0 #Ascites; bernoulli
		likelihood[12]=0 #Varices; bernoulli
		likelihood[13]=1 #Biliburn; continuous
		likelihood[14]=1 #ALK phosphate; continuous
		likelihood[15]=1 #SGOT; continuous
		likelihood[16]=1 #Albumin; continuous
		likelihood[17]=1 #protime; continuous
		likelihood[18]=0 #Histology: bernoulli
		out=df1.Output.unique()
		x.init(x,df1,likelihood,out,df1.shape[0],df1.shape[1]-1, ypos)
		x.accuracy(x,5)
		
		# x.fit(x)
		# acc = x.accuracy(x,5)
		# print("hepatitis data model accuracy: ", acc*100, "%")
		input_vector = [[51,1,1,1,1,1,2,2,2,2,2,2,2,1.00,78,58,4.6,52,1], [57,1,1,2,1,1,2,2,2,2,1,1,2,4.60,82,55,3.3,30,2]]
		# res = x.predict(x,input_vector)
		# output_res = [None] * len(res)
		# for i in range(len(res)):
		# 	if res[i] == 1:
		# 		output_res[i] = "Die"
		# 	elif res[i]==2:
		# 		output_res[i] = "Live"
		# print(output_res)


