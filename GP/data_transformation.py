class DataTransformation:

    def __init__(self):
        pass

    def transform_X(self, X):
        raise NotImplementedError()

    def transform_Y(self, Y):
        raise NotImplementedError()

    def untransform_X(self, X):
        raise NotImplementedError()

    def untransform_Y(self, Y):
        raise NotImplementedError()

    def untransform_Y_var(self, Yvar):
        raise NotImplementedError()


class IdentityTransformation:

    def __init__(self):
        pass

    def transform_X(self, X):
        return X

    def transform_Y(self, Y):
        return Y

    def untransform_X(self, X):
        return X

    def untransform_Y(self, Y):
        return Y

    def untransform_Y_var(self, Yvar):
        return Yvar

    @staticmethod
    def get_transformation(Y, X):
        return IdentityTransformation()


class MeanTransformation(object, DataTransformation):

    def __init__(self, mean):
        super(MeanTransformation, self).__init__()
        self.mean = mean

    def transform_X(self, X):
        return X

    def transform_Y(self, Y):
        return Y - self.mean

    def untransform_X(self, X):
        return X

    def untransform_Y(self, Y):
        return Y + self.mean

    def untransform_Y_var(self, Yvar):
        return Yvar

    @staticmethod
    def get_transformation(Y, X):
        return MeanTransformation(Y.mean(axis=0))
