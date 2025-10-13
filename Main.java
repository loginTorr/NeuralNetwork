import java.util.Random;
import java.text.DecimalFormat;
import java.util.Arrays;

class Network {
    private final Random rand = new Random();
    public final int num_layers;
    public final int[] sizes; //number of neurons in the respective layers, [2,3,1] for input:2, hidden:3, and final:1
    public final double[][][] biases = {
        {
            {0.1, -0.36, -0.31},
        },
        {
            {0.16, -0.46}   
        }
    }; 
    public final double[][][] weights = {
        {
            {-0.21, 0.72, -0.25, 1,
            -0.94, -0.41, -0.47, 0.63,
            0.15, 0.55, -0.49, -0.75},
        },
        {
            {0.76, 0.48, -0.73, 
            0.34, 0.89, -0.23}
        },
    }; 

    public Network(int[] sizes) {
        this.num_layers = 3;
        this.sizes = sizes.clone(); 
    }

    public double[] sigmoid(double[] z) {
        double[] result = new double[z.length];
        for (int i=0; i < z.length; i++) {
            result[i] = 1.0/(1.0 + Math.exp(-z[i]));   
        }
        return result;
    }

    public double[] sigmoidPrime(double[] z) {
        double[] sig = sigmoid(z);
        double[] result = new double[z.length];
        for (int i=0; i < z.length; i++) {
            result[i] = sig[i] * (1-sig[i]);
        }
        return result;
    }

    public double[] feedForward(double[] a) {
        
        //hidden layer
        double[] z1 = new double[biases[0][0].length]; 
        for (int i = 0; i < biases[0][0].length; i++) {
            double sum = 0.0;
            for (int j = 0; j < a.length; j++) {
                sum += weights[0][0][i * a.length + j] * a[j];
            }
            z1[i] = sum + biases[0][0][i];
        }
    
        //outer layer
        double[] a1 = sigmoid(z1);
        double[] z2 = new double[biases[1][0].length]; 
        
        for (int i = 0; i < biases[1][0].length; i++) {
            double sum = 0.0;
            for (int j = 0; j < a1.length; j++) {
                sum += weights[1][0][i * a1.length + j] * a1[j];
            }
            z2[i] = sum + biases[1][0][i]; 
        }
        
        return sigmoid(z2); 
    }

    public void SGD(double[][][] trainingData, int epochs, int miniBatchSize, double eta) {
        int n = trainingData.length;
        
        for (int epoch = 1; epoch < epochs; epoch++) {
            // Create mini-batches
            int numMiniBatches = n / miniBatchSize;
            for (int k = 0; k < numMiniBatches; k++) {
                double[][][] miniBatch = new double[miniBatchSize][2][];
                for (int i = 0; i < miniBatchSize; i++) {
                    miniBatch[i][0] = trainingData[k * miniBatchSize + i][0]; // input
                    miniBatch[i][1] = trainingData[k * miniBatchSize + i][1]; // output
                }
                updateMiniBatch(miniBatch, eta);
            }
            
            
            // print statement for last epoch
            if (epoch == epochs - 1) {
                System.out.println("Epoch 6 complete");
                for (int i = 0; i < trainingData.length; i++) {
                    double[] input = trainingData[i][0];
                    double[] output = feedForward(input);
                    System.out.println("Y" + (i+1) + ": " + Arrays.toString(output) );
                }
            }
        }
    }

    public void updateMiniBatch(double[][][] miniBatch, double eta) {
        // Initialize gradient accumulators for biases and weights with zeros
        double[][][] nablaB = new double[biases.length][1][];
        for (int i = 0; i < biases.length; i++) {
            nablaB[i][0] = new double[biases[i][0].length];
        }
        
        double[][][] nablaW = new double[weights.length][1][];
        for (int i = 0; i < weights.length; i++) {
            nablaW[i][0] = new double[weights[i][0].length];
        }
    
        for (int i = 0; i < miniBatch.length; i++) {
            double[] x = miniBatch[i][0]; // input
            double[] y = miniBatch[i][1]; // expected output
            
            BackpropResult result = backprop(x, y);
            double[][][] deltaNablaB = result.deltaNablaB;
            double[][][] deltaNablaW = result.deltaNablaW;
            
            // Add gradients
            for (int j = 0; j < nablaB.length; j++) {
                for (int k = 0; k < nablaB[j][0].length; k++) {
                    nablaB[j][0][k] += deltaNablaB[j][0][k];
                }
            }
            
            for (int j = 0; j < nablaW.length; j++) {
                for (int k = 0; k < nablaW[j][0].length; k++) {
                    nablaW[j][0][k] += deltaNablaW[j][0][k];
                }
            }
        }

        // Update weights and biases using gradients
        for (int j = 0; j < weights.length; j++) {
            for (int k = 0; k < weights[j][0].length; k++) {
                weights[j][0][k] -= (eta/miniBatch.length) * nablaW[j][0][k];
            }
        }

        for (int j = 0; j < biases.length; j++) {
            for (int k = 0; k < biases[j][0].length; k++) {
                biases[j][0][k] -= (eta/miniBatch.length) * nablaB[j][0][k];
            }
        }
    }

    public double[] costDerivative(double[] outputActivations, double[] y) {
        double[] result = new double[outputActivations.length];
        for (int i = 0; i < outputActivations.length; i++) {
            result[i] = outputActivations[i] - y[i];
        }
        return result;
    }

    class BackpropResult {
        public double[][][] deltaNablaB;
        public double[][][] deltaNablaW;
        
        public BackpropResult(double[][][] deltaNablaB, double[][][] deltaNablaW) {
            this.deltaNablaB = deltaNablaB;
            this.deltaNablaW = deltaNablaW;
        }
    }

    public BackpropResult backprop(double[] x, double[] y) {
        // Initialize gradient holders for 3D arrays
        double[][][] nablaB = new double[biases.length][1][];
        for (int i = 0; i < biases.length; i++) {
            nablaB[i][0] = new double[biases[i][0].length];
        }
        
        double[][][] nablaW = new double[weights.length][1][];
        for (int i = 0; i < weights.length; i++) {
            nablaW[i][0] = new double[weights[i][0].length];
        }

        // Forward pass
        double[][] activations = new double[num_layers][];
        activations[0] = x;
        
        // Store all z values for backprop
        double[][] zs = new double[num_layers-1][];
        
        // Hidden layer calculations - ensure precise calculation order
        double[] z1 = new double[biases[0][0].length];
        for (int i = 0; i < biases[0][0].length; i++) {
            double sum = 0.0;
            for (int j = 0; j < x.length; j++) {
                int weightIndex = i * x.length + j;
                // Use a more stable summation approach
                sum += weights[0][0][weightIndex] * x[j];
            }
            z1[i] = sum + biases[0][0][i];
        }
        zs[0] = z1;
        
        double[] a1 = sigmoid(z1);
        activations[1] = a1;
        
        // Output layer calculations - ensure precise calculation order
        double[] z2 = new double[biases[1][0].length];
        for (int i = 0; i < biases[1][0].length; i++) {
            double sum = 0.0;
            for (int j = 0; j < a1.length; j++) {
                int weightIndex = i * a1.length + j;
                // Use a more stable summation approach
                sum += weights[1][0][weightIndex] * a1[j];
            }
            z2[i] = sum + biases[1][0][i];
        }
        zs[1] = z2;
        
        double[] a2 = sigmoid(z2);
        activations[2] = a2;

        // Backward pass
        // Output layer error
        double[] delta = costDerivative(activations[2], y);
        double[] sp = sigmoidPrime(zs[1]);
        
        for (int i = 0; i < delta.length; i++) {
            delta[i] *= sp[i];
        }
        
        // Set nabla_b for output layer
        for (int i = 0; i < biases[1][0].length; i++) {
            nablaB[1][0][i] = delta[i];
        }
        
        // Set nabla_w for output layer
        for (int i = 0; i < biases[1][0].length; i++) {
            for (int j = 0; j < activations[1].length; j++) {
                // Fix the index calculation in the weights array
                int weightIndex = i * activations[1].length + j;
                nablaW[1][0][weightIndex] = delta[i] * activations[1][j];
            }
        }
        
        // Hidden layer error
        double[] hiddenDelta = new double[biases[0][0].length];
        for (int i = 0; i < hiddenDelta.length; i++) {
            double sum = 0.0;
            for (int j = 0; j < delta.length; j++) {
                // Fix the index calculation for backpropagating errors
                int weightIndex = j * activations[1].length + i;
                sum += weights[1][0][weightIndex] * delta[j];
            }
            sp = sigmoidPrime(zs[0]);
            hiddenDelta[i] = sum * sp[i];
        }
        
        // Set nabla_b for hidden layer
        for (int i = 0; i < biases[0][0].length; i++) {
            nablaB[0][0][i] = hiddenDelta[i];
        }
        
        // Set nabla_w for hidden layer
        for (int i = 0; i < biases[0][0].length; i++) {
            for (int j = 0; j < activations[0].length; j++) {
                // Fix the index calculation in the weights array
                int weightIndex = i * activations[0].length + j;
                nablaW[0][0][weightIndex] = hiddenDelta[i] * activations[0][j];
            }
        }

        return new BackpropResult(nablaB, nablaW);
    }

    public int argmax(double[] array) {
        int maxIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public int evaluate(double[][][] testData) {
        int correct = 0;
        for (int i = 0; i < testData.length; i++) {
            double[] input = testData[i][0];
            double[] expected = testData[i][1];
            double[] output = feedForward(input);
            
            if (argmax(output) == argmax(expected)) {
                correct++;
            }
        }
        return correct;
    }

    public static void printMatrix(double[] array) {
        System.out.println(Arrays.toString(array));
    }

    public static void printMatrix(double[][] matrix) {
        for (int i = 0; i < matrix.length; i++) {
            System.out.println(Arrays.toString(matrix[i]));
        }
    }
}

public class Main {
    public static void main(String[] args) {
        double[][][] trainingData = {
            {
                {0.0, 1.0, 0.0, 1.0}, {0.0, 1.0},  
            },
            {
                {1.0, 0.0, 1.0, 0.0}, {1.0, 0.0}
            },
            {
                {0.0, 0.0, 1.0, 1.0}, {0.0, 1.0}
            },
            {
                {1.0, 1.0, 0.0, 0.0}, {1.0, 0.0}
            },
        };
        
        Network network = new Network(new int[]{4, 3, 2});
        network.SGD(trainingData, 6, 2, 10.0);
    }
}