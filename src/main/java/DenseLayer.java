/**
 * A regular, fully connected neural network
 * It has the absolute minimum needed to work - no fancy activation functions or gradient descent optimizers
 *
 * @author Amos Decker
 *
 * @ersion 1.0
 */
package main.java;

import main.java.Utilities.Matrix;

import java.util.ArrayList;


public class DenseLayer
{
    int[] neuronsPerLayer;
    int numLayers;

    Matrix[] weights;
    Matrix[] biases;

    double stepSize;
    ArrayList<Matrix> outputs; // the outputs of each layer
    Matrix expectedOutputs;


    public DenseLayer(int[] neuronsPerLayer)
    {
        numLayers = neuronsPerLayer.length;
        this.neuronsPerLayer = neuronsPerLayer;

        weights = new Matrix[numLayers - 1];
        // initialize random weights
        for (int w = 0; w < this.numLayers - 1; w ++)
        {
            weights[w] = new Matrix( new double[neuronsPerLayer[w]][neuronsPerLayer[w + 1]] ).random();
        }


        biases = new Matrix[numLayers - 1];
        // initialize random biases
        for (int b = 1; b < numLayers; b++)
        {
            biases[b - 1] = new Matrix( new double[1][neuronsPerLayer[b]]).random();
        }

        stepSize = 0.01; // stepWithPrediction size is also called the alpha or learning rate

        outputs = new ArrayList<>();

    }

    /**
     * Changes the input and expected output
     * @param newInput
     * @param newExpectedOutput
     */
    public void setInputOutput(Matrix newInput, Matrix newExpectedOutput)
    {
        outputs.add(newInput); // the input acts as an output of a layer
        expectedOutputs = newExpectedOutput;
    }

    /**
     * forward pass through the net
     */
    public void feedforward()
    {
        ArrayList<Matrix> tempOut = new ArrayList<>();
        tempOut.add(outputs.get(0));
        outputs = tempOut;

        ArrayList<Matrix> weightedInputs = new ArrayList<>();
        for (int l = 0; l < numLayers - 1; l++ ) // subtract one because the input is already in outputs
        {
            outputs.add(
                    Matrix.sigmoidMat(outputs.get(l).dot(weights[l]).add(biases[l])));
        }

    }

    /**
     * gets the cross entropy loss gradient
     * @return CE loss gradient, a Matrix object
     */
    private Matrix getCostDeriv()
    {
        // expected output - the actual output
        return expectedOutputs.subtractMat(outputs.get(outputs.size() - 1));
    }

    /**
     * Figures out how much to adjust the weights by
     * @return weightGradients and biasGradients
     */
    public ArrayList<ArrayList<Matrix>> backprop()
    {
        ArrayList<Matrix> weightGradients = new ArrayList<>();
        ArrayList<Matrix> biasGradients = new ArrayList<>();

        Matrix costDeriv = getCostDeriv();

        Matrix penultError = outputs.get(outputs.size() - 2).T().dot(costDeriv);
        weightGradients.add(0, penultError); // put it at the beginning so that when all the errors are calculated the
        // first layer's error will be the first item in the list
        biasGradients.add(costDeriv);

        Matrix delta = costDeriv;

        int layerNum = outputs.size() - 3; // I already calculated the first two above (this is also the reason
        int weightIndex = weights.length - 1;
        // I add 2 for the weights)
        //System.out.println("layer num: " + layerNum);

        for (int l = 0; l < numLayers - 2; l ++)
        {
            delta = delta.dot(weights[weightIndex].T());
            delta = delta.elementMult(Matrix.sigmoidPrimeMat(outputs.get(layerNum + 1)));

            biasGradients.add(0, delta);

            weightGradients.add(0,
                    outputs.get(layerNum).T().dot(delta) // error calc
            );
            layerNum --;
            weightIndex --;
        }


        // return the weight gradients and bias gradients
        ArrayList<ArrayList<Matrix>> returnItems = new ArrayList<>();
        returnItems.add(weightGradients);
        returnItems.add(biasGradients); // the first item in here, the delta, will be used for backprop in LSTM1
        return returnItems;

    }

    /**
     * This is the learning part where the weights and biases are adjusted to better fit the pattern.
     Call this after backprop()

     @param weightGradients:  how much to adjust the weights by
     @param biasGradients: how much to adjust the biases by
     */
    public void adjustWeights(ArrayList<Matrix> weightGradients, ArrayList<Matrix> biasGradients)
    {
        for (int w = 0; w < weights.length; w ++)
        {
            weights[w] = weights[w].add(
                    weightGradients.get(w).scalarMultiply(stepSize)
            );

            biases[w] = biases[w].add(
                    biasGradients.get(w).scalarMultiply(stepSize)
            );
        }
    }

    public void train(Matrix input, Matrix expectedOutputs)
    {

        setInputOutput(input, expectedOutputs);

        int timesThrough = 30000;
        Matrix delta = null;
        for (int i = 0; i < timesThrough; i++)
        {

            feedforward();

            ArrayList<ArrayList<Matrix>> backpropReturn = backprop(); // return weightGradients and biasGradients
            delta = backpropReturn.get(1).get(0); // can be used for backprop in LSTM1
            adjustWeights(backpropReturn.get(0), backpropReturn.get(1));
        }
        System.out.println("Delta: \n");
        delta.printMat();
        System.out.println("\n");
        System.out.println("OUTPUT: \n");
        outputs.get(outputs.size() - 1).printMat();
    }
}
