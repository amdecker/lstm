package main.java;


import main.java.Utilities.Matrix;
import java.util.ArrayList;
import java.util.Scanner;

import static main.java.Utilities.Utility.*;

/**
 * Runner3 class for LSTM and where I do some testing
 * to increase memory for intellij, go to Run - Edit Configurations and put -Xmx3000m in VM options to get 3000 mb of memory
 *
 * Goes with LSTM
 * @author Amos Decker
 * @version 12/28/17
 */
public class Runner
{
    // I got this stuff from https://stackoverflow.com/questions/5762491/how-to-print-color-in-console-using-system-out-println
    public static final String ANSI_RESET = "\u001B[0m";
    public static final String ANSI_BLACK = "\u001B[30m";
    public static final String ANSI_RED = "\u001B[31m";
    public static final String ANSI_GREEN = "\u001B[32m";
    public static final String ANSI_YELLOW = "\u001B[33m";
    public static final String ANSI_BLUE = "\u001B[34m";
    public static final String ANSI_PURPLE = "\u001B[35m";
    public static final String ANSI_CYAN = "\u001B[36m";
    public static final String ANSI_WHITE = "\u001B[37m";

    public static void main(String[] args) throws java.io.IOException, InterruptedException
    {
        Scanner sc = new Scanner(System.in);
        print("Would you like to see the intro? (y/n): ");
        if (sc.nextLine().equals("y"))
        {
            sleepy("Hello and welcome to", 200);
            sleepy("...", 400);
            println("");
            sleepy("the program known as", 200);
            sleepy("...", 400);
            println("");
            println(ANSI_CYAN + "Fill in the " + ANSI_RESET + ANSI_RED + "[REDACTED]" + ANSI_RESET);
            sleepy("  ", 500);
            println("");

            print(ANSI_PURPLE);
            println("Doing cool animation...  ");
            loadingRotation(3, 300);
            println("");
            println("DONE!");
            print(ANSI_RESET);
        }

        int intChoice = -1;
        while(intChoice != 0)
        {
            intChoice = -1;
            println("\n");
            println("What would" + ANSI_GREEN + " you " + ANSI_RESET + "like to do:");
            println("0. exit");
            println("1. " + ANSI_RED + "Train and test ABCDE" + ANSI_RESET);
            println("2. " + ANSI_YELLOW + "the senate cia report" + ANSI_RESET);

            // get a valid choice from the user
            while (intChoice == -1)
            {
                System.out.print("I want to do option: ");

                String choice = sc.nextLine();
                try
                {
                    intChoice = Integer.parseInt(choice);
                    if (intChoice != 0 && intChoice != 1 && intChoice != 2)
                    {
                        intChoice = -1;
                    }
                } catch (NumberFormatException e)
                {
                }
                if (intChoice == -1)
                {
                    System.out.println("Please enter a valid number");
                }
            }
            println("");

            switch (intChoice)
            {
                case 0: break; // the while loop above makes it so that this will only run if 0 is not chosen
                case 1:
                    println("An " + ANSI_GREEN + "epoch" + ANSI_RESET + " is a full pass through the training data, which" +
                            " in this case are the letters A, B, C, D, and E.");

                    int numEpochs = -1;
                    while (numEpochs == -1)
                    {
                        print("How many " + ANSI_GREEN + "epochs " + ANSI_RESET + "would you like to train the net for? " +
                                "\n(a few thousand will make it work perfectly -- and don't say 0): ");
                        String num = sc.nextLine();
                        try
                        {
                            numEpochs = Integer.parseInt(num);
                            if (numEpochs == 0)
                            {
                                numEpochs = -1;
                            }
                        } catch (NumberFormatException e)
                        { }
                        if (numEpochs == -1)
                        {
                            System.out.println(ANSI_RESET + "Please enter a valid number");
                        }
                    }
                    println("");
                    println(ANSI_CYAN + "The numbers you will see are probabilities. The highest number is " +
                            "the highest probability of being next" + ANSI_RESET);
                    testABCDE(numEpochs);
                    break;
                case 2:
                    // TEST 1 layer
                    loadWeightsTestOneLayer("senate/senate1", 512);
                    System.out.println("\n");
                    System.out.println("finished predictions");
                    break;
                default: break;
            }
        }

        System.out.println(ANSI_BLUE + "Have a nice day! \uD83C\uDF05 " + ANSI_YELLOW + "☀" + "  \uD83C\uDF08");
    }


    /**
     * This uses the letters ABCDE and shows what each letter should be, the actual prediction, and the prediction matrix
     *
     * It is a super simple task, but it works, 30,000 epochs should do it
     *
     * @param epochs the number of times through the training set (ABCDE)
     */
    public static void testABCDE(int epochs)
    {
        Matrix[] inputs = new Matrix[]{
                new Matrix(new double[][]{{1, 0, 0, 0, 0}}).T(), // A
                new Matrix(new double[][]{{0, 1, 0, 0, 0}}).T(), // B
                new Matrix(new double[][]{{0, 0, 1, 0, 0}}).T(), // C
                new Matrix(new double[][]{{0, 0, 0, 1, 0}}).T(), // D
                new Matrix(new double[][]{{0, 0, 0, 0, 1}}).T()}; // E

        Matrix[] expectedOutputs = new Matrix[]{
                new Matrix(new double[][]{{0, 1, 0, 0, 0}}).T(), // B
                new Matrix(new double[][]{{0, 0, 1, 0, 0}}).T(), // C
                new Matrix(new double[][]{{0, 0, 0, 1, 0}}).T(), // D
                new Matrix(new double[][]{{0, 0, 0, 0, 1}}).T(), // E
                new Matrix(new double[][]{{1, 0, 0, 0, 0}}).T()}; // A

        int numThings = expectedOutputs.length;

        LSTM lstm = new LSTM(expectedOutputs[0].shape[0], 5);
        lstm.setInput(inputs);
        lstm.setExpectedOutputs(expectedOutputs);
        lstm.initializeVarsBasedOnExpOut(expectedOutputs.length);
        lstm.initializeWeightsAndBiases(inputs[0].shape[0]);

        println("TRAINING...");
        for (int i = 0; i < epochs; i++) // epochs
        {
            if (i % (Double.max(epochs / 10, 1)) == 0) { System.out.print("epoch: " + i + "\n"); }

            for (int in = 0; in < numThings; in++) // for each training example/time stepWithPrediction
            {
                lstm.stepWithPrediction(in); // make a prediction as to what the next character will be
            }
            Matrix[][][] gradients = lstm.backprop(null, false); // figure out what adjustments
                                                                                        // will make it more accurate
            lstm.adjustWeights(lstm.adamOptimizer(gradients)); // make those adjustments
        }

        String word = "ABCDE";
        String nextPredicted = "BCDEA";
        println("");
        for (int n = 0; n < numThings; n++) // for each training example/time stepWithPrediction
        {
            //System.out.print("time stepWithPrediction: " + in + "\n");
            Matrix pred = lstm.predictions[n];

            // get the maxindex
            int maxIndex = -1;
            double max = Double.MIN_VALUE;
            for (int r = 0; r < pred.shape[0]; r++)
            {
                if (pred.matrix[r][0] > max)
                {
                    max = pred.matrix[r][0];
                    maxIndex = r;
                }
            }
            System.out.println("Should be " + nextPredicted.charAt(n) + " is " + word.charAt(maxIndex));
            System.out.println("maximum index: " + maxIndex);
            pred.printMat();
            System.out.println("\n");
        }

        // this is where some seed values are given and it predicts the next one, which is then used to predict the next etc
        System.out.println("NEW INPUTS");
        lstm.predictions = new Matrix[expectedOutputs.length]; // reset predictions
        // WITH NEW INPUTS hel
        Matrix[] newInputs = new Matrix[]{
                new Matrix(new double[][]{{1, 0, 0, 0, 0}}).T(),
                new Matrix(new double[][]{{0, 1, 0, 0, 0}}).T(),
                new Matrix(new double[][]{{0, 0, 1, 0, 0}}).T()};

        lstm.setInput(newInputs);
        // build up internal states and make predictions
        for (int in = 0; in < newInputs.length; in++) // for each training example/time stepWithPrediction
        {
            lstm.stepWithPrediction(in);
        }

        word = "ABCDE";
        nextPredicted = "BCDEA";
        for (int n = 0; n < newInputs.length; n++) // for each training example/time stepWithPrediction
        {
            Matrix pred = lstm.predictions[n];

            // get the max index
            int maxIndex = -1;
            double max = Double.MIN_VALUE;
            for (int r = 0; r < pred.shape[0]; r++)
            {
                if (pred.matrix[r][0] > max)
                {
                    max = pred.matrix[r][0];
                    maxIndex = r;
                }
            }
            System.out.println("current index: " + n);
            System.out.println("predicted index: " + maxIndex);
            System.out.println("Given: " + word.charAt(n));
            System.out.println("Should be " + nextPredicted.charAt(n) + " is " + word.charAt(maxIndex));
            System.out.println("");
            pred.printMat();
        }
    }

    /**
     * loads preprocessed inputs and then trains the network
     * One layer of LSTM
     * @param folder the folder of where to save the weights
     * @param savedFile the file with the saved encodings
     * @param hiddenDim the hidden dimension
     *
     * @throws java.io.IOException
     */
    public static void trainLSTMOneLayer(String folder, String savedFile, int hiddenDim) throws java.io.IOException
    {
        String pathToProject = System.getProperty("user.dir") + "/src";

        System.out.println("One layer loading inputs...");
        Matrix[] inputs = Matrix.load(pathToProject + "/main/java/" + savedFile);
        System.out.println("One layer Done loading inputs");

        Matrix[] expectedOutputs = new Matrix[inputs.length - 1];
        for (int i = 0; i < inputs.length - 1; i++)
        {
            expectedOutputs[i] = inputs[i + 1];
        }

        int trainingAmnt = 1; // how much to backprop over at one time

        LSTM lstm = new LSTM(inputs[0].shape[0], hiddenDim);
        lstm.initializeWeightsAndBiases(inputs[0].shape[0]);
        lstm.initializeVarsBasedOnExpOut(trainingAmnt);

        int numThings = expectedOutputs.length;

        System.out.println("numThings: " + numThings);


        for (int i = 0; i < 1000; i++) // epochs
        {
            if (i % 50 == 0)
            {
                System.out.println("1 layer epoch: " + i);
            }
            for (int b = 0; b < numThings / (trainingAmnt) + 1; b++)
            {
                //println("b: " + b);
                Matrix[] partInputs          = new Matrix[trainingAmnt];
                Matrix[] partExpectedOutputs = new Matrix[trainingAmnt];

                // if the last part can't be put into its own full section, don't include any of it
                if (b * trainingAmnt + trainingAmnt > numThings) { break; }

                for (int x = 0; x < trainingAmnt; x++)
                {

                    partInputs[x] = inputs[b * trainingAmnt + x];
                    partExpectedOutputs[x] = expectedOutputs[b * trainingAmnt + x];
                }

                lstm.setInput(partInputs);
                lstm.setExpectedOutputs(partExpectedOutputs);

                int numInputs = partInputs.length;
                for (int in = 0; in < trainingAmnt; in++) // for each training example/time stepWithPrediction
                {
                    lstm.stepWithPrediction(in);
                }

                Matrix[][][] gradients = lstm.backprop(null, false);
                lstm.adjustWeights(lstm.adamOptimizer(gradients));
            }

            clearWeightData(folder);
            for (Matrix W : lstm.W)
            {
                W.save(pathToProject + "/main/java/weights/" + folder + "/Wweights.txt");
            }
            for (Matrix U : lstm.U)
            {
                U.save(pathToProject + "/main/java/weights/" + folder + "/Uweights.txt");
            }
            for (Matrix bias : lstm.biases)
            {
                bias.save(pathToProject + "/main/java/weights/" + folder + "/biases.txt");
            }
            lstm.V.save             (pathToProject + "/main/java/weights/" + folder + "/Vweights.txt");
            lstm.predictionBias.save(pathToProject + "/main/java/weights/" + folder + "/predBiases.txt");
            lstm.prevOutput.save    (pathToProject + "/main/java/weights/" + folder + "/prevOutput.txt");
            lstm.prevMemory.save    (pathToProject + "/main/java/weights/" + folder + "/prevMemory.txt");
        }
    }

    /**
     * runs with the saved weights
     * @param folder the folder of where to save the weights
     * @param hiddenDim the hidden dimension
     * @throws java.io.IOException
     */
    public static void loadWeightsTestOneLayer(String folder, int hiddenDim) throws java.io.IOException
    {
        String pathToProject = System.getProperty("user.dir");

        int size = 200;
        Matrix[] newInput = new Matrix[size + 1];
        Preprocess p = new Preprocess("");

        Scanner sc = new Scanner(System.in);
        System.out.print("Please enter starter characters (if you enter more than " + size + ", it will be cut off): ");
        StringBuilder startWord = new StringBuilder();

        startWord.append(sc.nextLine());

        // clean and make on-hot matrices of the starting chars
        ArrayList<Matrix> starters = p.cleanEncode(startWord);
        for (int i = 0; i < Integer.min(size, starters.size()); i++)
        {
            newInput[i] = starters.get(i);
        }

        char[] acceptedChars = new char[]{'’', 'ʻ', '“', '”', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', ' ', '\t', '\n', '\r', '\f'};


        LSTM lstm = new LSTM(acceptedChars.length, hiddenDim);
        lstm.initializeWeightsAndBiases(acceptedChars.length);
        lstm.setInput(newInput);
        lstm.initializeVarsBasedOnExpOut(newInput.length);

        lstm.W      = Matrix.load(pathToProject + "/src/main/java/weights/" + folder + "/Wweights.txt");

        lstm.U      = Matrix.load(pathToProject + "/src/main/java/weights/" + folder + "/Uweights.txt");
        lstm.biases = Matrix.load(pathToProject + "/src/main/java/weights/" + folder + "/biases.txt");
        // load returns an array, but there is only one of these
        lstm.V              = Matrix.load(pathToProject + "/src/main/java/weights/" + folder + "/Vweights.txt")[0];
        lstm.predictionBias = Matrix.load(pathToProject + "/src/main/java/weights/" + folder + "/predBiases.txt")[0];

        int lengthOutput = size; // cant be greater than size of newInput array, includes the starting chars

        // this builds up the prevMemory and prevOutput without making any predictions
        int timesThrough = 0;
        for (int s = 0; s < startWord.length() - 1; s++)
        {
            lstm.stepNoPrediction(s);
            timesThrough++;
        }

        println("\nPredicted chars: ");
        // this starts at the end of the start word where we can start making predictions based on the start word
        for (int t = startWord.length() - 1; t < lengthOutput; t++)
        {
            lstm.setInput(newInput);
            Matrix prediction = lstm.stepWithPrediction(t);
            int maxIndex = prediction.getMaxIndex()[0]; // the rows for max index are the only one that matters (b/c all matrices are nx1)
            newInput[timesThrough + 1] = p.makeOneHotMatrix(maxIndex, acceptedChars.length); // make the max 1 and the rest 0

            if (acceptedChars[maxIndex] == '\n') { print("\\n"); } // \n take up a lot of space, so dont actually print a new line
            else { System.out.print(acceptedChars[maxIndex]); }
            timesThrough++;
        }
    }

    /**
     * loads preprocessed inputs and then trains the network
     * 2 layers of LSTM
     * @param folder the folder of where to save the weights
     * @param savedFile the file with the saved encodings
     * @param hiddenDim the hidden dimension size
     * @throws java.io.IOException
     */
    public static void trainLSTMTwoLayers(String folder, String savedFile, int hiddenDim) throws java.io.IOException
    {
        String pathToProject = System.getProperty("user.dir") + "/src";

        System.out.println("loading inputs...");
        Matrix[] inputs = Matrix.load(pathToProject + "/main/java/" + savedFile);
        System.out.println("Done loading inputs");

        Matrix[] expectedOutputs = new Matrix[inputs.length - 1];
        for (int i = 0; i < inputs.length - 1; i++) { expectedOutputs[i] = inputs[i + 1]; }

        int trainingAmnt = 50; // how much to backprop over at one time

        /*
        lstm0 is the first layer. It takes the one-hot encoded inputs and produces an output
        lstm1 is the second and 3rd layer. It takes the output from the first layer and then makes a prediction
        using a single fully connected (aka dense) neural network layer.
         */
        LSTM lstm0 = new LSTM(inputs[0].shape[0], hiddenDim);
        lstm0.initializeWeightsAndBiases(inputs[0].shape[0]);
        lstm0.initializeVarsBasedOnExpOut(trainingAmnt);

        LSTM lstm1 = new LSTM(inputs[0].shape[0], hiddenDim);
        lstm1.initializeWeightsAndBiases(inputs[0].shape[0]);
        lstm1.initializeVarsBasedOnExpOut(trainingAmnt);


        int numThings = expectedOutputs.length;

        System.out.println("numThings: " + numThings);
        for (int i = 0; i < 1000; i++) // epochs
        {
            if (i % 50 == 0) { System.out.print("epoch: " + i + "\n"); }

            for (int b = 0; b < numThings / (trainingAmnt) + 1; b++) // for each group of inputs and outputs
            {
                // if the last part can't be put into its own full section, don't include any of it
                if (b * trainingAmnt + trainingAmnt > numThings) { break; }

                Matrix[] lstm0PartInputs     = new Matrix[trainingAmnt];
                Matrix[] lstm1PartInputs     = new Matrix[trainingAmnt];
                Matrix[] lstm1partExpectedOutputs = new Matrix[trainingAmnt];

                // get the inputs and expected outputs for this group
                for (int x = 0; x < trainingAmnt; x++)
                {
                    lstm0PartInputs[x]          = inputs[b * trainingAmnt + x];
                    lstm1partExpectedOutputs[x] = expectedOutputs[b * trainingAmnt + x];
                }

                lstm0.setInput(lstm0PartInputs);
                lstm1.setExpectedOutputs(lstm1partExpectedOutputs);

                // for each training example/time get a prediction
                for (int in = 0; in < trainingAmnt; in++)
                {
                    lstm1PartInputs[in] = lstm0.stepWithPrediction(in); // first lstm layer
                    lstm1.setInput(lstm1PartInputs);
                    lstm1.stepWithPrediction(in); // second lstm layer and 3rd layer (the fully connected one)
                }
                Matrix[][][] lstm1Gradients = lstm1.backprop(null, true);
                lstm1.adjustWeights(lstm1.adamOptimizer(lstm1Gradients));

                Matrix[][][] lstm0Gradients = lstm0.backprop(lstm1.delta_inputs, false);
                lstm0.adjustWeights(lstm0.adamOptimizer(lstm0Gradients));
            }

            // Save weights and biases
            String[] layersString = new String[] {"/lstm0", "/lstm1"};
            LSTM[] layers         = new LSTM[] {lstm0, lstm1};
            for (int l = 0; l < layersString.length; l++)
            {
                clearWeightData(folder + layersString[l]); // this gets rid of the data previously stored there
                for (Matrix W : layers[l].W)
                {
                    W.save(pathToProject + "/main/java/weights/" + folder + layersString[l] + "/Wweights.txt");
                }
                for (Matrix U : layers[l].U)
                {
                    U.save(pathToProject + "/main/java/weights/" + folder + layersString[l] + "/Uweights.txt");
                }
                for (Matrix bias : layers[l].biases)
                {
                    bias.save(pathToProject + "/main/java/weights/" + folder + layersString[l] + "/biases.txt");
                }
                layers[l].V.save(pathToProject + "/main/java/weights/" + folder + layersString[l] + "/Vweights.txt");
                layers[l].predictionBias.save(pathToProject + "/main/java/weights/" + folder + layersString[l] + "/predBiases.txt");
            }
        }
    }

    /**
     * runs with the saved weights for 2 layers
     * @param folder the folder of where to save the weights
     * @param hiddenDim the hidden dimension
     * @throws java.io.IOException
     */
    public static void loadWeightsTestTwoLayers(String folder, int hiddenDim) throws java.io.IOException
    {
        String pathToProject = System.getProperty("user.dir") + "/src";

        Preprocess p = new Preprocess("");

        Scanner sc = new Scanner(System.in);
        System.out.print("Please enter starter characters: ");
        StringBuilder startWord = new StringBuilder();

        startWord.append(sc.nextLine());

        // clean and make on-hot matrices of the starting chars
        ArrayList<Matrix> starters = p.cleanEncode(startWord);

        int starterSize = starters.size();
        int numPredictions = 10;
        starterSize += numPredictions;

        Matrix[] newInput    = new Matrix[starterSize + numPredictions];
        Matrix[] lstm1Inputs = new Matrix[starterSize + numPredictions];

        for (int i = 0; i < starters.size(); i++) { newInput[i] = starters.get(i); }


        char[] acceptedChars = new char[]{'’', 'ʻ', '“', '”', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', ' ', '\t', '\n', '\r', '\f'};

        LSTM lstm0 = new LSTM(newInput[0].shape[0], hiddenDim);
        lstm0.initializeWeightsAndBiases(newInput[0].shape[0]);
        lstm0.initializeVarsBasedOnExpOut(newInput.length);
        lstm0.setInput(newInput);


        LSTM lstm1 = new LSTM(newInput[0].shape[0], hiddenDim);
        lstm1.initializeWeightsAndBiases(lstm0.hiddenDim); // the output of lstm0 will be of dimension hidden x 1
        lstm1.initializeVarsBasedOnExpOut(newInput.length);  // for more info on dimensions see comment in LSTM5()

        String[] layersString = new String[] {"/lstm0", "/lstm1"};
        LSTM[] layers         = new LSTM[] {lstm0, lstm1};
        for (int l = 0; l < layersString.length; l++)
        {
            layers[l].W = Matrix.load(pathToProject + "/main/java/weights/" + folder + layersString[l] + "/Wweights.txt");
            layers[l].U = Matrix.load(pathToProject + "/main/java/weights/" + folder + layersString[l] + "/Uweights.txt");
            layers[l].biases = Matrix.load(pathToProject + "/main/java/weights/" + folder + layersString[l] + "/biases.txt");
            layers[l].V = Matrix.load(pathToProject + "/main/java/weights/" + folder + layersString[l] + "/Vweights.txt")[0];
            layers[l].predictionBias = Matrix.load(pathToProject + "/main/java/weights/" + folder + layersString[l] + "/predBiases.txt")[0];
        }

        int lengthOutput = starterSize; // cant be greater than size of newInput array, includes the starting chars

        // this builds up the prevMemory and prevOutput without making any final predictions
        int timesThrough = 0;
        for (int s = 0; s < startWord.length() - 1; s++)
        {
            //println("s: " + s);
            lstm1Inputs[s] = lstm0.stepWithPrediction(s);
            lstm1.setInput(lstm1Inputs);
            lstm1.stepNoPrediction(s);
            timesThrough++;
        }

        println("timesThroughBeforePredictions: " + timesThrough);

        System.out.println("\n Predicted chars: ");
        // this starts at the end of the start word where we can start making predictions based on the start word
        for (int t = startWord.length() - 1; t < lengthOutput; t++)
        {
            lstm0.setInput(newInput);
            lstm1Inputs[t] = lstm0.stepWithPrediction(t);

            lstm1.setInput(lstm1Inputs);
            Matrix prediction = lstm1.stepWithPrediction(t);

            int maxIndex = prediction.getMaxIndex()[0]; // the rows for max index are the only one that matters (b/c all matrices are nx1)
            newInput[timesThrough + 1] = p.makeOneHotMatrix(maxIndex, acceptedChars.length); // make the max 1 and the rest 0
            System.out.print(acceptedChars[maxIndex]);
            timesThrough++;
        }
    }

    /**
     * clears the content in the weight files
     * @param folder the folder to clear the weight data
     * @throws java.io.IOException
     */
    public static void clearWeightData(String folder) throws java.io.IOException
    {
        String pathToProject = System.getProperty("user.dir")  + "/src";
        writeFile(pathToProject + "/main/java/weights/" + folder + "/biases.txt", "", false);
        writeFile(pathToProject + "/main/java/weights/" + folder + "/predBiases.txt", "", false);
        writeFile(pathToProject + "/main/java/weights/" + folder + "/Uweights.txt", "", false);
        writeFile(pathToProject + "/main/java/weights/" + folder + "/Vweights.txt", "", false);
        writeFile(pathToProject + "/main/java/weights/" + folder + "/Wweights.txt", "", false);
        writeFile(pathToProject + "/main/java/weights/" + folder + "/prevMemory.txt", "", false);
        writeFile(pathToProject + "/main/java/weights/" + folder + "/prevOutput.txt", "", false);

        //System.out.println(pathToProject + folder + " CLEARED!");
    }
}

