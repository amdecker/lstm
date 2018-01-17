package main.java;

import main.java.Utilities.Matrix;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;

/**
 * Given an input text file, it cleans it up and can make one-hot encodings of
 * the characters and what the expected output should be for the LSTM
 *
 * @author Amos Decker
 * @version 0.0.1 -- 12/28/17
 *
 */
public class Preprocess
{
    File inputFile;

    int totalAcceptedChars = 0;
    int[] indices; // this holds the indices of which accepted char it is
    // the number of accepted chars will never be beyond the original string's length

    public Preprocess(String fileName)
    {
        inputFile = new File(fileName);
    }

    public StringBuilder readContents() throws FileNotFoundException
    {
        Scanner reader = new Scanner(inputFile);
        StringBuilder contents = new StringBuilder();
        int count = 0;
        while(reader.hasNextLine())
        {
            if (count % 1000 == 0)
            {
                System.out.println("reading line " + count);
            }
            contents.append(reader.nextLine() + "\n");

            count ++;
        }

        return contents;
    }

    /**
     * No, this function does not make attractive matrices, it uses one-hot encoding to store the characters
     *
     * So if you were encoding the letters h and i from the word
     * "hi", the matrix for the h would be [1, 0] while the one for the i
     * would be [0, 1]
     *
     * @param index : which spot to put the 1
     * @param total : how many spots there are
     * @return a new matrix object from the array that is made
     */
    public Matrix makeOneHotMatrix(int index, int total)
    {
        double[][] encoded = new double[total][1]; // array is initialized to all zeros
        encoded[index][0] = 1; // put the 1 at the right spot
        return new Matrix(encoded);
    }

    /**
     * Removes unwanted characters and returns the encoded matrices
     *
     * @param contents from readContents
     */
    public ArrayList<Matrix> cleanEncode(StringBuilder contents)
    {
        // I got the characters from python's string.printable and then added some of my own
        char[] acceptedChars = new char[]{'’', 'ʻ', '“', '”', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', ' ', '\t', '\n', '\r', '\f'};
        totalAcceptedChars = acceptedChars.length;

        ArrayList<Matrix> encoded = new ArrayList<>();

        for (int c = 0; c < contents.length(); c++)
        {
            char currentChar = contents.charAt(c);

            boolean isAccepted = false;
            // loop through the accepted chars to make sure that this is one of them
            for (int ac = 0; ac < acceptedChars.length; ac++)
            {
                if (currentChar == acceptedChars[ac])
                {
                    isAccepted = true;
                    encoded.add(makeOneHotMatrix(ac, totalAcceptedChars));
                }
            }
            // if it is not accepted, print it
            if (!isAccepted) { System.out.println(c + ": " + currentChar); }
        }
        return encoded;
    }
}
