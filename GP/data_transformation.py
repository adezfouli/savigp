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


class MinTransformation(object, DataTransformation):

    def __init__(self, min, max, offset):
        super(MinTransformation, self).__init__()
        self.min = min
        self.max = max
        self.offset = offset

    def transform_X(self, X):
        return X

    def transform_Y(self, Y):
        return (Y-self.min).astype('float')/(self.max-self.min) - self.offset

    def untransform_X(self, X):
        return X

    def untransform_Y(self, Y):
        return (Y+self.offset)*(self.max-self.min) + self.min

    def untransform_Y_var(self, Yvar):
        return Yvar * (self.max-self.min) ** 2

    @staticmethod
    def get_transformation(Y, X):
        return MinTransformation(Y.min(), Y.max(), 0.5)
