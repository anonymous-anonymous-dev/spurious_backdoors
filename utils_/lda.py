import numpy as np

class LDA2Class:
    """
    Two-class Fisher Linear Discriminant Analysis (LDA)
    Learn direction w that maximally separates X1 and X2.
    """

    def __init__(self, eps=1e-8):
        self.eps = eps
        self.w = None        # projection direction
        self.m1 = None       # mean class 1
        self.m2 = None       # mean class 2
    
    def fit(self, X1, X2):
        """
        Train LDA on two distributions.
        Inputs:
            X1: (N1, D)
            X2: (N2, D)
        """
        X1 = np.asarray(X1)
        X2 = np.asarray(X2)
        D = X1.shape[1]

        # class means
        self.m1 = X1.mean(axis=0)
        self.m2 = X2.mean(axis=0)

        # within-class scatter matrix Sw = S1 + S2
        S1 = np.cov(X1, rowvar=False)
        S2 = np.cov(X2, rowvar=False)
        Sw = S1 + S2 + self.eps * np.eye(D)

        # best LDA direction: w ∝ Sw^{-1} (m1 - m2)
        mean_diff = self.m1 - self.m2
        self.w = np.linalg.solve(Sw, mean_diff)
        self.w /= np.linalg.norm(self.w)  # normalize direction
        return self

    def transform(self, X):
        """
        Project unseen inputs X onto learned LDA direction.
        X: (M, D)
        Returns: (M,)  1D projection values
        """
        if self.w is None:
            raise ValueError("Model is not fit yet.")
        return X @ self.w

    def predict_class(self, X):
        """
        Classify unseen samples: class 1 or class 2.
        Decision boundary = midpoint of projected means.
        """
        proj = self.transform(X)
        # projected means
        m1p = self.m1 @ self.w
        m2p = self.m2 @ self.w
        threshold = 0.5 * (m1p + m2p)
        return (proj > threshold).astype(int)

    def score(self, X, y):
        """
        Compute classification accuracy on labeled data.
        X: (M, D)
        y: (M,)  labels (0 or 1)
        Returns: accuracy (float)
        """
        y_pred = self.predict_class(X)
        return np.mean(y_pred == y)
    
    