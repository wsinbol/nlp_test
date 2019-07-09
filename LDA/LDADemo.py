from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import make_multilabel_classification
import numpy as np
# This produces a feature matrix of token counts, similar to what
# CountVectorizer would produce on text.
X, _ = make_multilabel_classification(random_state=0)
X = np.array([
	[0,0,0,4,5],
	[1,2,3,0,0],
	[0,0,0,1,1],
	])

lda = LatentDirichletAllocation(n_components=2,
    random_state=0)

lda.fit(X)
# get topics for some given samples
print(lda.transform([[0,0,0,1,1]]))