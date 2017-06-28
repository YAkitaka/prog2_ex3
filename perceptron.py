

class Perceptron(object):

    def __init__(self, learning_rate):

        self.weights = [1,1]
        self.lr = learning_rate # ordinary 0<lr<1
        self.bias = 1.0

    def prediction(self, x):

        z =(self.weights[0] * x[0] + self.weights[1]*x[1] + self.bias)
        activation = get_activation(z)
        
        return activation

    def train(self,x,y):
        
        prediction = self.prediction(x)

        for i in range(len(x)):
            update_weight = self.weights[i] + self.lr * (y - prediction) * x[i]
            self.weights[i] = update_weight
            
        update_bias = self.bias + self.lr*(y-prediction)
        self.bias = update_bias
        

def get_activation(z):

    return int(z > 0)


def train_file():

    file = open("/Users/masaaki/pythonPractice/training_data.txt","r")

    for line in file:

        x = []
        t = 0

        data = line.split()
        object1 = Perceptron(0.5)

        x.append(float(data[0]))
        x.append(float(data[1]))
        t += float(data[2])

        object1.train(x, t)

        print("x[0]={},x[1]={},y={},trained_weights={},trained_bias={}".\

              format(x[0],x[1],t,object1.weights, object1.bias))

    file.close()


def main():

    train_file()


if __name__ == '__main__':

    main()
    






        

