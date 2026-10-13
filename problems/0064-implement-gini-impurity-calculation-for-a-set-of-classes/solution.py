import numpy as np

def gini_impurity(y):
	"""
	Calculate Gini Impurity for a list of class labels.

	:param y: List of class labels
	:return: Gini Impurity rounded to three decimal places
	"""
	yd = np.array(y)
	classes = np.unique(y)
	val = 1.0
	for c in classes:
		val -= np.mean(yd == c) ** 2
	return round(val,3)