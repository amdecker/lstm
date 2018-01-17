package main.java;
import main.java.Utilities.Matrix;

/**
 * Class for making an LSTM (long short-term memory), a type of RNN (recurrent neural network)
 *
 * The equations are based ones from here: https://arxiv.org/pdf/1503.04069.pdf but I have not implemented the peephole connections
 * and here: https://medium.com/@aidangomez/let-s-do-this-f9b699de31d9
 *
 * this gives a good basic explanation of an LSTM: https://medium.com/mlreview/understanding-lstm-and-its-diagrams-37e2f46f1714
 *
 * And these two posts are fantastic: https://colah.github.io/posts/2014-07-NLP-RNNs-Representations/ and
 * https://karpathy.github.io/2015/05/21/rnn-effectiveness/
 *
 * @author Amos Decker
 * @version 1.3.0 --- 1/15/18 --- cleaned up the code, but functions the same as 1.2.0
 */

public class LSTM
{
    int inputDim;
    int hiddenDim;
    int desiredOutputDim;

    Matrix[] inputs;
    Matrix[] expectedOutputs;
    Matrix[] outputs; // the outputs of the LSTM1 time steps
    Matrix[] predictions; // the predictions of the LSTM1 time steps NOTE: different from outputs


    Matrix prevOutput;
    Matrix prevMemory;

    // W matrices act on inputs
    Matrix[] W;

    // U matrices act on the memory
    Matrix U[];

    // biases vectors are the bias for each weighted sum
    Matrix biases[];

    // transforms the output
    Matrix V;
    Matrix predictionBias;

    double stepSize = .01; // was .001

    // holds every output from the forget (or input or output... etc) gate
    Matrix[] forgetGateReturns;
    Matrix[] inputGateReturns;
    Matrix[] newMemoryGateReturns;
    Matrix[] outputGateReturns;
    Matrix[] outputLayerReturns;
    Matrix[] candidateValues;

    Matrix[] delta_inputs; //This is only needed if there are multiple layers of LSTMs. It stores the delta for the input in backprop

    //Set up things needed for the adam optimizer
    double beta1 = 0.9;
    double beta2 = .999;
    double eps = Math.pow(10, -8);
    Matrix[][] W_prevMovement1;
    Matrix[][] W_prevMovement2;
    Matrix[][] U_prevMovement1;
    Matrix[][] U_prevMovement2;

    Matrix[][] V_prevMovement1;
    Matrix[][] V_prevMovement2;

    Matrix[][][] all_prevMovement1;
    Matrix[][][] all_prevMovement2;

    int timeStep = 0; // for adam optimizer

    /**
     * sets up some variables that use these dimensions:
     * @param desiredOutputDim what dimension the output should be
     * @param hiddenDim what the hidden dimension should be
     */
    public LSTM(int desiredOutputDim, int hiddenDim)
    {
        /*  Indices for weight matrix arrays:
            0 Forget
            1 input
            2 new memory
            3 output

            h_{t-1} is the previous output
            x is the input
            b is the bias
            * is matrix mult
            e* is element wise mult

            Some of the Matrix dimensions:
                W: hidden x inputDim
                U: hidden x hidden
             bias: hidden x 1
                V: inputDim x hidden

                 input: inputDim x 1
                output: hidden x 1
            prediction: inputDim x 1

            delta_input: inputDim x 1
            WGradients_allTimeSteps: hidden x inputDim
            UGradients_allTimeSteps: hidden x hidden
            costDeriv: inputDim x 1

         */

        this.hiddenDim   = hiddenDim; // was 2, hopefully 512 will be better
        this.desiredOutputDim = desiredOutputDim;

        prevOutput = new Matrix(new double[hiddenDim][1]);
        prevMemory = new Matrix(new double[hiddenDim][1]);

    }

    /**
     * reset the input to new values and change any variables associated with it
     * @param newInput
     */
    public void setInput(Matrix[] newInput)
    {
        inputs = newInput;
        inputDim = inputs[0].shape[0]; // the number of rows
    }

    /**
     * Set the expected outputs and reset all the things that depend on it (those things are used when training)
     * @param newExpectedOutputs the name says it all
     */
    public void setExpectedOutputs(Matrix[] newExpectedOutputs)
    {
        expectedOutputs = newExpectedOutputs;
    }

    /**
     * this should be called 1 time to initialize some of the variables that depend on the length of the expected output
     * @param lengthExpOutput this is used to make arrays that are the length of the number of examples
     */
    public void initializeVarsBasedOnExpOut(int lengthExpOutput)
    {
        outputs     = new Matrix[lengthExpOutput];
        predictions = new Matrix[lengthExpOutput];

        // initialize lists and add a blank Matrix to the beginning or end (of the ones that need it)
        // so that there will always be a something in the right spot when doing backprop
        forgetGateReturns = new Matrix[lengthExpOutput + 1];
        forgetGateReturns[forgetGateReturns.length - 1] = new Matrix(new double[hiddenDim][1]);
        inputGateReturns        = new Matrix[lengthExpOutput];
        newMemoryGateReturns    = new Matrix[lengthExpOutput + 1]; // need to add 1 to whatever index you want
        newMemoryGateReturns[0] = new Matrix(new double[hiddenDim][1]);
        outputGateReturns       = new Matrix[lengthExpOutput];
        outputLayerReturns      = new Matrix[lengthExpOutput];
        candidateValues         = new Matrix[lengthExpOutput];

        delta_inputs = new Matrix[lengthExpOutput];

        /*
        Set up stuff for adam optimizer
         */
        // there are 4 types of W and U weights and each type of weight is there for all the timesteps - the lengthExpOutput
        // the timesteps are the sequence of inputs, so the first timestep of hello would be the matrix of 'h'
        W_prevMovement1 = Matrix.fill2dMatArray(new int[] {lengthExpOutput, 4}, new double[hiddenDim][inputDim]);

        W_prevMovement2 = W_prevMovement1.clone();
        U_prevMovement1 = Matrix.fill2dMatArray(new int[] {lengthExpOutput, 4}, new double[hiddenDim][hiddenDim]);

        U_prevMovement2 = U_prevMovement1.clone();

        // there is only 1 type of V weight (b/c it only is used for the prediction, and there is only one prediction
        // for each time step
        V_prevMovement1 = Matrix.fill2dMatArray(new int[] {lengthExpOutput, 1}, new double[desiredOutputDim][hiddenDim]);
        V_prevMovement2 = V_prevMovement1.clone();

        all_prevMovement1 = new Matrix[][][]{
                W_prevMovement1,
                U_prevMovement1,
                V_prevMovement1};

        all_prevMovement2 = new Matrix[][][]{
                W_prevMovement2,
                U_prevMovement2,
                V_prevMovement2};
    }

    /**
     * This sets up the weights and biases
     * Call this after setting the expected outputs
     *
     * @param inputDim, if setInput() has not been called you can supply the inputDim, if it has been called, give it -1
     */
    public void initializeWeightsAndBiases(int inputDim)
    {
        // the input dim might have already been set in setInput(), so you can just send in -1
        if (inputDim != -1) { this.inputDim = inputDim; }

        W = new Matrix[]{
                new Matrix(new double[hiddenDim][inputDim]).random(),
                new Matrix(new double[hiddenDim][inputDim]).random(),
                new Matrix(new double[hiddenDim][inputDim]).random(),
                new Matrix(new double[hiddenDim][inputDim]).random()};

        U = new Matrix[]{
                new Matrix(new double[hiddenDim][hiddenDim]).random(),
                new Matrix(new double[hiddenDim][hiddenDim]).random(),
                new Matrix(new double[hiddenDim][hiddenDim]).random(),
                new Matrix(new double[hiddenDim][hiddenDim]).random()};

        biases = new Matrix[]{
                new Matrix(new double[hiddenDim][1]).random(),
                new Matrix(new double[hiddenDim][1]).random(),
                new Matrix(new double[hiddenDim][1]).random(),
                new Matrix(new double[hiddenDim][1]).random()};

        V = new Matrix(new double[desiredOutputDim][hiddenDim]).random();
        predictionBias = new Matrix(new double[desiredOutputDim][1]).random();
    }

    /**
     * Decides which values to keep and which to forget by putting the weighted input, weighted hidden, and a bias
     * though the sigmoid function - the values that become close to 1 are "remembered" and the values that become
     * close to 0 are "forgotten"
     *
     * @param input: the one-hot encoded input vector
     * @return forgetReturn the matrix of what to remember and what to forget
     */
    public Matrix forgetGate(Matrix input)
    {/*
        h_{t-1} is the previous output
        x is the input
        b is the bias

        f_t = W * x + U * h_{t-1} + b
        f_t = sig(f_t)
        */
        Matrix weightedInput = W[0].dot(input);
        Matrix weightedPrevOut = U[0].dot(prevOutput);

        Matrix weightedForget = weightedInput.add(weightedPrevOut).add(biases[0]);
        Matrix forgetReturn = Matrix.sigmoidMat(weightedForget);
        
        return forgetReturn;
    }

    /**
     * Decides which values of the input to use
     * @param input: the one-hot encoded input vector
     * @return inputGateReturn : the modified input matrix
     */
    public Matrix inputGate(Matrix input)
    {
        /*
         i_t = W * x + U * h_{t-1} + b
         i_t = sig(i_t)
         */
        Matrix weightedInput = W[1].dot(input);
        Matrix weightedPrevOut = U[1].dot(prevOutput);

        Matrix weightedInputCalc = weightedInput.add(weightedPrevOut).add(biases[1]);
        Matrix inputGateReturn = Matrix.sigmoidMat(weightedInputCalc);

        return inputGateReturn;
    }

    /**
     * Make new internal cell state - the memory of the lstm
     *
     * @param input the one-hot encoded input vector
     * @param inputGateOut the output of the inputGate()
     * @param forgetGateOut the output of the forgetGate()
     * @param index the index of the time step, used to save the things calculated in this method into arrays
     * @return the new cell state
     */
    public Matrix newMemoryGate(Matrix input, Matrix inputGateOut, Matrix forgetGateOut, int index)
    {
        /*
        z is the possible values to update the cell state
        z_t = W * x + U * h_{t-1} + b
        z_t = tanh(z)

        C_t = z_t e* i_t + C_{t-1} e* f_t //i_t is from inputGate(), f_t is from forgetGate()
         */

        // calculate candidate values for the new memory
        Matrix weightedInput = W[2].dot(input);
        Matrix weightedPrevOut = U[2].dot(prevOutput);

        Matrix beforeTanhCandidateVals = weightedInput.add(weightedPrevOut).add(biases[2]);
        Matrix candidateVals = Matrix.tanhMat(beforeTanhCandidateVals);
        candidateValues[index] = candidateVals;

        // calculate new memory
        Matrix newMemoryGateOut = forgetGateOut.elementMult(prevMemory).add(
                inputGateOut.elementMult(candidateVals));

        prevMemory = newMemoryGateOut; // update prevMemory to its new value, prevMemory is used in line above
        newMemoryGateReturns[index + 1] = newMemoryGateOut; // add to array
        return newMemoryGateOut;
    }

    /**
     * decides what to include in the output
     * @param input the one-hot encoded input vector
     * @return newOutput the output at this time step
     */
    public Matrix outputGate(Matrix input)
    {
        /*
            o_t = W * x + U * h_{t-1} + b
            o_t = sig(o_t)
         */
         Matrix beforeActivationPartOfOutput = W[3].dot(input).add(
            U[3].dot(prevOutput)).add(biases[3]);

        Matrix outputGateret = Matrix.sigmoidMat(beforeActivationPartOfOutput);

        return outputGateret;
    }

    /**
     * gets the final output at this time step - note, this is NOT the prediction, it is the output of the LSTM layer
     * the prediction is after the output of this method goes through the fully connected layer which is implemented
     * in makePrediction()
     *
     * @param outputGateRet from outputGate()
     * @param currentMemory from the newMemoryGate()
     * @return newOutput the output at this time step
     */
    public Matrix outputLayer(Matrix outputGateRet, Matrix currentMemory)
    {
        /*
            h_t = tanh(C_t) e* o_t
         */

        Matrix newOutput = Matrix.tanhMat(currentMemory).elementMult(outputGateRet);
        return newOutput;
    }

    /**
     * Makes a prediction for the next character (or whatever the sequence is of) using the softmax activation function
     * softmax gives a probability distribution as output, so the prediction would be the item in this matrix with the
     * highest number
     *
     * @param outputLayerRet
     * @return the prediction matrix
     */
    public Matrix makePrediction(Matrix outputLayerRet)
    {
        return Matrix.softmax(V.dot(outputLayerRet).add(predictionBias));
    }

    /**
     * gets the cross entropy loss gradient using softmax,
     *
     * as this https://www.ics.uci.edu/~pjsadows/notes.pdf says:
     * "Note that this is the same formula as in the case with the logistic output units! The values themselves
     * will be different, because the predictions y will take on different values depending on whether the
     * output is logistic or softmax, but this is an elegant simplification."
     *
     * This means the CE deriv is the same whether you are using the sigmoid function or the softmax function, which
     * is pretty cool and makes things easier
     *
     * @param index which inputs and epected outputs to look at
     * @return CE loss gradient, a Matrix object
     */
    private Matrix getCostDeriv(int index)
    {
        // expected output - the actual output
        return expectedOutputs[index].subtractMat(predictions[index]);
    }

    /**
     * does all parts of the feedforward step except doing the step for getting a prediction
     *
     * This could be useful when trying to build up the internal cell state, but don't need to waste computational
     * time getting a prediction (like when using a seed as a starting point to predict the next characters)
     *
     * @param inputIndex the index of the inputs to use
     * @return layerOutput the output of the lstm layer, not the same as the prediction
     */
    public Matrix stepNoPrediction(int inputIndex)
    {
        Matrix input = inputs[inputIndex];

        Matrix forgotten        = forgetGate(input);
        forgetGateReturns[inputIndex] = forgotten;

        Matrix modifiedInput    = inputGate(input);
        inputGateReturns[inputIndex] = modifiedInput;

        Matrix newMemory        = newMemoryGate(input, modifiedInput, forgotten, inputIndex);
        Matrix outputGateRet    = outputGate(input);
        outputGateReturns[inputIndex] = outputGateRet;

        Matrix layerOutput      = outputLayer(outputGateRet, newMemory);
        outputs[inputIndex] = layerOutput;

        return layerOutput;
    }

    /**
     * Perform the calculations needed to feed data forwards
     *
     * @param inputIndex the index for inputs of what input to use (the time step)
     * @return prediction, the prediction of the net
     */
    public Matrix stepWithPrediction(int inputIndex)
    {
        Matrix input = inputs[inputIndex];

        // selectively forgets some of the inputs that are not important
        Matrix forgotten        = forgetGate(input);
        forgetGateReturns[inputIndex] = forgotten;

        Matrix modifiedInput    = inputGate(input);
        inputGateReturns[inputIndex] = modifiedInput;

        Matrix newMemory        = newMemoryGate(input, modifiedInput, forgotten, inputIndex);
        Matrix outputGateRet    = outputGate(input);

        outputGateReturns[inputIndex] = outputGateRet;

        // the output of the lstm
        Matrix layerOutput      = outputLayer(outputGateRet, newMemory);
        outputs[inputIndex] = layerOutput;

        // this part is the regular dense neural net part that interprets the output of the lstm layer
        Matrix prediction       = makePrediction(layerOutput);
        predictions[inputIndex] = prediction;

        return prediction;
    }

    /**
     * Through all time steps not just one
     * backpropagate the error to get the weight gradients (how much to adjust the weights by)
     *
     * There is some psuedo-code above many of the calculations to make them slightly easier to read
     *
     * @param costDerivs if null is sent, the costDeriv is calculated in the loop
     * @param calculateInputDelta if true it, well, calculates the input delta
     * @return the weight gradients and bias gradients
     */
    public Matrix[][][] backprop(Matrix[] costDerivs, boolean calculateInputDelta)
    {
        /*
        * In the variable names:
        * next means t + 1
        * prev means t - 1 (even though the "prev" delta outputs will be calculated after
        *                   the one being calculated at time t)
        */

        // matrices for delta_output computation
        Matrix next_delta_CandidateVals       = new Matrix(new double[hiddenDim][1]); // when you make a new matrix it is initialized to zeros
        Matrix next_delta_InputGateReturn     = new Matrix(new double[hiddenDim][1]);
        Matrix next_delta_ForgetReturn        = new Matrix(new double[hiddenDim][1]);
        Matrix next_delta_outputGateReturn    = new Matrix(new double[hiddenDim][1]);

        // matrices for U weight update gradient
        Matrix next_delta_beforeTanhCandidateVals       = new Matrix(new double[hiddenDim][1]);
        Matrix next_delta_weightedInputGateCalc         = new Matrix(new double[hiddenDim][1]);
        Matrix next_delta_weightedForget                = new Matrix(new double[hiddenDim][1]);
        Matrix next_delta_beforeActivationPartOfOutput  = new Matrix(new double[hiddenDim][1]);

        // the first time through, this will be zeros, but after that it will have other values in it
        Matrix next_delta_newMemoryGateOut = new Matrix(new double[hiddenDim][1]);

        Matrix[][] WGradients_allTimeSteps = new Matrix[inputs.length][4];
        Matrix[][] UGradients_allTimeSteps = new Matrix[inputs.length][4];
        Matrix[][] bGradients_allTimeSteps = new Matrix[inputs.length][4];

        // these two need to be 2d arrays so that I can put all the gradients into one 3d array at the end for the return
        // they only actually need to be 1d arrays, so the first spot will be the only one used
        Matrix[][] VGradients_allTimeSteps        = new Matrix[inputs.length][1];
        Matrix[][] predBiasGradients_allTimeSteps = new Matrix[inputs.length][1];

        forgetGateReturns[forgetGateReturns.length - 1] = new Matrix(new double[hiddenDim][1]); //so that it can do t + 1 for the first time through

        for (int t = inputs.length - 1; t >= 0; t--)
        {
            Matrix costDeriv;
            if (costDerivs != null) { costDeriv = costDerivs[t]; }
            else {  costDeriv = getCostDeriv(t); }

            //delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            Matrix delta_t = V.T().dot(costDeriv).elementMult((outputs[t].elementMult(outputs[t]))
                    .scalarAdd(-1.0).scalarMultiply(-1)); // this is doing 1 - something, it is weird, but it works

            /*
            d_lo_t = ∆_t +
            U[2].T() * d_z_{t + 1}) +
            U[1].T() * d_i_{t + 1}) +
            U[0].T() * d_f_{t + 1}) +
            U[3].T() * d_o_{t + 1})
            */
            Matrix delta_layerOutput = delta_t.add(
                    U[2].T().dot(next_delta_CandidateVals)   ).add(
                    U[1].T().dot(next_delta_InputGateReturn) ).add(
                    U[0].T().dot(next_delta_ForgetReturn)    ).add(
                    U[3].T().dot(next_delta_outputGateReturn));

            /*
            d_o_t = d_lo_t e* tanh(c_t) e* sigPrime(o_t)
             */
            Matrix delta_outputGateRet = delta_layerOutput.elementMult(
                    Matrix.tanhMat(newMemoryGateReturns[t + 1])).elementMult( // newmemoryGateReturns is t + 1 to keep it at t
                    // b/c all of them are shifted b/c of the zeros at the beginning
                    Matrix.sigmoidPrimeMat(outputGateReturns[t]));

            /*
            d_c_t = d_lo_t e* o_t e* tanhPrime(c_t) + d_c_{t + 1} e* f_{t + 1}
             */
            Matrix delta_newMemoryGateOut = delta_layerOutput.elementMult(
                    outputGateReturns[t]).elementMult(
                    Matrix.tanhPrimeMat(newMemoryGateReturns[t + 1])).add(
                        next_delta_newMemoryGateOut.elementMult(
                        forgetGateReturns[t + 1]
            ));
            next_delta_newMemoryGateOut = delta_newMemoryGateOut; // update to the new value

            /*
            d_f_t = d_c_t e* c_{t - 1} e* sigPrime(f_t)
             */
            Matrix delta_forgetGateOut = delta_newMemoryGateOut.elementMult(
                    newMemoryGateReturns[t + 1 - 1]).elementMult( // the + 1 - 1 is for clarity of what I am doing
                            // t-1 is the previous new memory gate return, but all of them are shifted at the beginning,
                            // so you need to add one
                    Matrix.sigmoidPrimeMat(forgetGateReturns[t]));

            /*
            d_i_t = d_c_t e* z_t e* sigPrime(i_t)
             */
            Matrix delta_inputGateOut = delta_newMemoryGateOut.elementMult(
                    candidateValues[t]).elementMult(
                    Matrix.sigmoidPrimeMat(inputGateReturns[t]));

            /*
            d_z_t = d_c_t e* i_t e* tanhPrime(z_t)
             */
            Matrix delta_candidateVals = delta_newMemoryGateOut.elementMult(
                    inputGateReturns[t]).elementMult(
                    Matrix.tanhPrimeMat(candidateValues[t]));

            /*
            This is only needed if there are multiple layers of LSTMs. It gets the delta for the input
            d_x =
            W[2].T() * d_z_t) +
            W[1].T() * d_i_t) +
            W[0].T() * d_f_t) +
            W[3].T() * d_o_t)
             */
            if (calculateInputDelta)
            {
                delta_inputs[t] =
                        W[2].T().dot(delta_candidateVals).add(
                                W[1].T().dot(delta_inputGateOut)).add(
                                W[0].T().dot(delta_forgetGateOut)).add(
                                W[3].T().dot(delta_outputGateRet));
            }

            // calculate weight (and bias) gradients
                // W weights
            WGradients_allTimeSteps[t][2] = delta_candidateVals.outerProduct(inputs[t]);
            WGradients_allTimeSteps[t][1] = delta_inputGateOut.outerProduct(inputs[t]);
            WGradients_allTimeSteps[t][0] = delta_forgetGateOut.outerProduct(inputs[t]);
            WGradients_allTimeSteps[t][3] = delta_outputGateRet.outerProduct(inputs[t]);
                // U weights
            UGradients_allTimeSteps[t][2] = next_delta_beforeTanhCandidateVals.outerProduct(outputs[t]);
            UGradients_allTimeSteps[t][1] = next_delta_weightedInputGateCalc.outerProduct(outputs[t]);
            UGradients_allTimeSteps[t][0] = next_delta_weightedForget.outerProduct(outputs[t]);
            UGradients_allTimeSteps[t][3] = next_delta_beforeActivationPartOfOutput.outerProduct(outputs[t]);
                // biases
            bGradients_allTimeSteps[t][2] = delta_candidateVals;
            bGradients_allTimeSteps[t][1] = delta_inputGateOut;
            bGradients_allTimeSteps[t][0] = delta_forgetGateOut;
            bGradients_allTimeSteps[t][3] = delta_outputGateRet;
                // prediction weight (V) and bias
            VGradients_allTimeSteps[t][0]        = costDeriv.outerProduct(outputs[t]);
            predBiasGradients_allTimeSteps[t][0] = costDeriv;

            next_delta_CandidateVals = delta_candidateVals;
            next_delta_InputGateReturn = delta_inputGateOut;
            next_delta_ForgetReturn = delta_forgetGateOut;
            next_delta_outputGateReturn = delta_outputGateRet;
        }

        Matrix[][][] gradients = new Matrix[][][]{
                WGradients_allTimeSteps,
                UGradients_allTimeSteps,
                bGradients_allTimeSteps,
                VGradients_allTimeSteps,
                predBiasGradients_allTimeSteps};

        return gradients;
    }

    /**
     * An optimization of gradient descent
     * adam = "Adaptive Moment Estimation"
     *
     *The authors of the paper suggest a very low alpha, or step size (they say to use  0.001)
     * Here is the paper that describes it, with pseudo-code on page 2: https://arxiv.org/pdf/1412.6980.pdf
     *
     * @param weightGradients the gradient to optimize of a specific weight like W[0]
     */
    public Matrix[][][] adamOptimizer(Matrix[][][] weightGradients)
    {
        timeStep++;

        Matrix[][] WGradients = weightGradients[0];
        Matrix[][] UGradients = weightGradients[1];
        Matrix[][] VGradient  = weightGradients[3];

        Matrix[][][] justWeightGradients = new Matrix[][][]{
                WGradients,
                UGradients,
                VGradient};

        // for each type of weight (W, U, V)
        for (int w = 0; w < justWeightGradients.length; w++)
        {
            // for each type of that weight (W[0], etc)
            for (int ow = 0; ow < justWeightGradients[w].length; ow++)
            {
                Matrix[] thisWeightGrad = justWeightGradients[w][ow];
                Matrix[] movement1 = new Matrix[thisWeightGrad.length];
                Matrix[] movement2 = new Matrix[thisWeightGrad.length];
                Matrix[] weightAdjustments = new Matrix[thisWeightGrad.length];

                // for each gradient (from each time step)
                for (int g = 0; g < thisWeightGrad.length; g++)
                {
                    // movement1[g] = beta1 * prev_movement1[g] + (1 - beta1) * weight_gradients[g]
                    all_prevMovement1[w][ow][g] = all_prevMovement1[w][ow][g];

                    movement1[g] = all_prevMovement1[w][ow][g].scalarMultiply(beta1).add(
                            thisWeightGrad[g].scalarMultiply(1 - beta1));

                    //movement2 = beta2 * prev_movement2[g] + (1 - beta2) * (weight_gradients[g]^2))
                    movement2[g] = all_prevMovement2[w][ow][g].scalarMultiply(beta2).add(
                            thisWeightGrad[g].elementMult(thisWeightGrad[g]).scalarMultiply(1 - beta2));

                    Matrix biasCorrectedM1 = movement1[g].scalarDivide(1 - Math.pow(beta1, timeStep));
                    Matrix biasCorrectedM2 = movement2[g].scalarDivide(1 - Math.pow(beta2, timeStep));

                    weightAdjustments[g] = biasCorrectedM1.elementDivide(Matrix.sqrt(biasCorrectedM2).scalarAdd(eps));
                }

                all_prevMovement1[w][ow] = movement1;
                all_prevMovement2[w][ow] = movement2;

                justWeightGradients[w][ow] = weightAdjustments;
            }
        }
        // keep the bias grads, but change the weight grads
        Matrix[][][] newWeightGrads = weightGradients;
        newWeightGrads[0] = justWeightGradients[0];
        newWeightGrads[1] = justWeightGradients[1];
        newWeightGrads[3] = justWeightGradients[2];

        return newWeightGrads;
    }

    /**
     * Adjusts the weights based off of the gradients calculated in backprop()
     * This is where the "learning" that happened in backprop() is used to update the weights
     *
     * @param weightGradients from backprop()
     */
    public void adjustWeights(Matrix[][][] weightGradients)
    {
        // for each time step
        for(int t = 0; t < inputs.length; t++)
        {
            Matrix[] WGradients = weightGradients[0][t];
            Matrix[] UGradients = weightGradients[1][t];
            Matrix[] bGradients = weightGradients[2][t];

            // there is only one V and one prediction bias for each time step
            Matrix VGradient         = weightGradients[3][t][0];
            Matrix predBiasGradient = weightGradients[4][t][0];
            V = V.add(VGradient.scalarMultiply(stepSize));
            predictionBias = predictionBias.add(predBiasGradient.scalarMultiply(stepSize));

            // for type weight/bias for the different gates
            for(int w = 0; w < W.length; w++) // there are 4 of each weight/bias per time step
            {
                W[w] = W[w].add(WGradients[w].scalarMultiply(stepSize));
                U[w] = U[w].add(UGradients[w].scalarMultiply(stepSize));
                biases[w] = biases[w].add(bGradients[w].scalarMultiply(stepSize));
            }
        }
    }
}



