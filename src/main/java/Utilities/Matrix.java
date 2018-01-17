package main.java.Utilities;

import java.util.Arrays;
import java.util.ArrayList;
import java.io.FileWriter;
import java.util.Scanner;
import java.io.File;

/**
 *
 * Operations on matrices
 *
 * Now more object oriented! With a ton of other stuff
 * and uses double instead of float
 *
 * @author Caroline Zeng and Amos Decker
 * @version 2.11.0
 */

public class Matrix
{

    public double[][] matrix; // the thing that stores the actual values of the matrix

    public int[] shape; // #rows, #cols like [2, 3] would be 2 rows and 3 columns

    /**
     * makes a Matrix object
     *
     * if you give an array like
     * new double[][]{
     *  {1, 2, 3},
     *  {4, 5, 6}};
     * a matrix will be created with one of the rows as {1, 2, 3} and the other row as {4, 5, 6}
     *
     * @param mat a 2D array of doubles that is the matrix
     */
    public Matrix(double[][] mat)
    {
        double[][] d = new double[][] { {1, 2, 3}, {1, 2}};
        matrix = mat;

        shape = new int[]{mat.length, mat[0].length};

        // make sure all of the rows are of the same length
        int prevNumInRow = -1;
        for (int r = 0; r < shape[0] - 1; r++)
        {
            if (mat[r].length != mat[r + 1].length)
            {
                throw( new RuntimeException("Can't have matrix with rows of different sizes:" +
                        "\n Rows " + r + " and " + (r + 1) + " don't match: " + mat[r].length + " != " + mat[r + 1].length));
            }
        }
    }

    /**
     * Makes a matrix of the same size as the current matrix with random values between 0 and 1
     * TODO add normal random option
     * @return the randomized matrix
     */
    public Matrix random()
    {
        double[][] random = new double[shape[0]][shape[1]];

        // for each row and each column, put a random number in
        for (int r = 0; r < shape[0]; r ++)
        {
            for (int c = 0; c < shape[1]; c ++)
            {
                random[r][c] = Math.random();
            }
        }

        return new Matrix(random);
    }

    /**
     * Makes a matrix of the same size as the current matrix with all values as 0
     *
     * @return the 0 filled matrix
     */
    public Matrix zeros()
    {
        double[][] zero = new double[shape[0]][shape[1]];

        // for each row and each column, put a random number in
        for (int r = 0; r < shape[0]; r ++)
        {
            for (int c = 0; c < shape[1]; c ++)
            {
                zero[r][c] = 0.0;
            }
        }
        return new Matrix(zero);
    }

    /**
    * Dot product
    * now more object oriented, and made sum a double (before it was an int, so it could not handle doubles)
     * Multiply 2 matrices, given their dimensions allow them to be multiplied together
     * Usage: mat1.dot(mat2)
     * this.matrix is the left hand AxB matrix
     * @param mat2  right hand BxC matrix
     * @return      AxC matrix
     */
    public Matrix dot(Matrix mat2)
    {
        if (this.shape[1] != mat2.shape[0])
        {
            throw new RuntimeException(
                    "Cannot multiply shapes together: Cannot multiply (" +
                            shape[0] + "x" + shape[1] + ") with ("
                            + mat2.shape[0] + "x" + mat2.shape[1] + ")");
        }

        double[][] ret = new double[this.shape[0]][mat2.shape[1]];

        for (int r = 0; r < this.shape[0]; r++)
        {
            for (int c = 0; c < mat2.shape[1]; c++)
            {
                double sum     = 0;
                double[] row = this.matrix[r];
                double[] col = mat2.getCol(c);

                for (int i = 0; i < row.length; i++)
                {
                    sum += (row[i] * col[i]);
                }

                ret[r][c] = sum;
            }
        }
        return new Matrix(ret);
    }

    /**
     * Outer product of two matrices, as long as the first is  m x 1  and the second is  n x 1
     *
     * according to wikipedia,
     * "The outer product u ⊗ v is equivalent to a matrix multiplication uv^T, provided that u is represented
     * as a m × 1 column vector and v as a n × 1 column vector" (https://en.wikipedia.org/wiki/Outer_product)
     *
     * @param mat2 the matrix you are multiplying the other by
     * @return a matrix object, the outer product of mat1 and mat2 --- mat1.dot(mat2.T())
     */
    public Matrix outerProduct(Matrix mat2)
    {
        if (this.shape[1] != 1 || mat2.shape[1] != 1)
        {
            throw new RuntimeException("Cannot multiply shapes that are not m x 1 together: \n" +
                    "Cannot multiply (" + shape[0] + "x" + shape[1] + ") with " +
                    "(" + mat2.shape[0] + "x" + mat2.shape[1] + ")");
        }
        return this.dot(mat2.T());
    }

    /**
    * Usage: mat1.add(mat2)
    * if mat1 is of shape AxB
    * @param mat2 needs to be of shape AxB or Ax1 or 1xB
    * @return a 
    */
    public Matrix add(Matrix mat2)
    {
        if (this.shape[0] == 1 && this.shape[1] == 1)
        {
            return mat2.scalarAdd(this.matrix[0][0]);
        }
        if(mat2.shape[0] == 1 && mat2.shape[1] == 1)
        {
            return this.scalarAdd(mat2.matrix[0][0]);
        }

        boolean sameRows = this.shape[0] == mat2.shape[0];
        boolean sameCols = this.shape[1] == mat2.shape[1];

        boolean canAdd = sameRows && sameCols; // can add if same dimensions
        if (!canAdd)
            {
                // is good if either rows or columns doesn't match, but either (but not both) is 1
                if (!sameRows && sameCols)
                {
                    canAdd = this.shape[0] == 1 || mat2.shape[0] == 1;
                }
                else if (sameRows && !sameCols)
                {
                    canAdd = this.shape[1] == 1 || mat2.shape[1] == 1;
                }
            }
        // if you cannot add them, throw an error
        if (!canAdd)
        {
            throw new RuntimeException(
                    "Cannot add shapes together: Cannot add (" +
                            shape[0] + "x" + shape[1] + ") with ("
                            + mat2.shape[0] + "x" + mat2.shape[1] + ")");
        }


        // make a new matrix, sum, with the dimensions of sum being the max of the dimensions of the matrices being added together
        Matrix sum = new Matrix(new double[Math.max(this.shape[0], mat2.shape[0])][Math.max(this.shape[1], mat2.shape[1])]);

        int mat1RowIndex = 0;
        int mat1ColIndex = 0;

        int mat2ColIndex = 0;
        int mat2RowIndex = 0;

        for (int r = 0; r < sum.shape[0]; r++)
        {
            mat1ColIndex = 0;
            mat2ColIndex = 0;
            for (int c = 0; c < sum.shape[1]; c++)
            {
                sum.matrix[r][c] =
                        this.matrix[mat1RowIndex][mat1ColIndex] +
                        mat2.matrix[mat2RowIndex][mat2ColIndex];

                if (mat1ColIndex < this.shape[1] - 1) mat1ColIndex ++;
                if (mat2ColIndex < mat2.shape[1] - 1) mat2ColIndex ++;
            }
            if (mat1RowIndex < this.shape[0] - 1) mat1RowIndex ++;
            if (mat2RowIndex < mat2.shape[0] - 1) mat2RowIndex ++;
        }
        return sum;
    }

    /**
     * Subtracts one matrix from another, need to be same size or have dim 1
     *
     * The logic is the same as add, but instead it subtracts
     * usage: mat1.subtractMat(mat2)
     * @param mat2
     * @return
     */
    public Matrix subtractMat(Matrix mat2)
    {
        boolean sameRows = this.shape[0] == mat2.shape[0];
        boolean sameCols = this.shape[1] == mat2.shape[1];

        boolean canSubtract = sameRows && sameCols; // can add if same dimensions
        if (!canSubtract)
        {
            // is good if either rows or columns doesn't match, but either (but not both) is 1
            if (!sameRows && sameCols)
            {
                canSubtract = this.shape[0] == 1 || mat2.shape[0] == 1;
            }
            else if (sameRows && !sameCols)
            {
                canSubtract = this.shape[1] == 1 || mat2.shape[1] == 1;
            }
        }
        // if you cannot subtract them, throw an error
        if (!canSubtract)
        {
            throw new RuntimeException(
                    "Cannot subtract shapes: Cannot subtract (" +
                            shape[0] + "x" + shape[1] + ") and ("
                            + mat2.shape[0] + "x" + mat2.shape[1] + ")");
        }


        // make a new matrix, subtracted, with the dimensions being the max of the possible dimensions
        Matrix subtracted = new Matrix(new double[Math.max(this.shape[0], mat2.shape[0])][Math.max(this.shape[1], mat2.shape[1])]);

        int mat1RowIndex = 0;
        int mat1ColIndex = 0;

        int mat2ColIndex = 0;
        int mat2RowIndex = 0;

        for (int r = 0; r < subtracted.shape[0]; r++)
        {
            mat1ColIndex = 0;
            mat2ColIndex = 0;
            for (int c = 0; c < subtracted.shape[1]; c++)
            {
                // the actual subtracting part
                subtracted.matrix[r][c] =
                        this.matrix[mat1RowIndex][mat1ColIndex] -
                                mat2.matrix[mat2RowIndex][mat2ColIndex];

                // don't increase the index if it will result if the matrix doesn't have that much in it
                if (mat1ColIndex < this.shape[1] - 1) mat1ColIndex ++;
                if (mat2ColIndex < mat2.shape[1] - 1) mat2ColIndex ++;
            }
            if (mat1RowIndex < this.shape[0] - 1) mat1RowIndex ++;
            if (mat2RowIndex < mat2.shape[0] - 1) mat2RowIndex ++;
        }
        return subtracted;
    }

    /**
     * Gets the transpose of a matrix
     * Essentially flips matrix along its diagonal
     *
     * Amos changed 12/16/17 to make it work with non square matrices (matrices where the #rows != #columns)
     *
     * @return      transpose of matrix, a matrix object
     */
    public Matrix transpose()
    {
        int dim       = matrix.length; // # rows
        int otherDim  = matrix[0].length; // # columns
        // the original matrix would be [dim][otherDim], so it should be flipped when transposing
        double[][] transposed = new double[otherDim][dim];

        for (int i = 0; i < otherDim; i++)
        {
            for (int j = 0; j < dim; j++)
            {
                transposed[i][j] = matrix[j][i];
            }
        }
        return new Matrix(transposed);
    }

    /**
     * Shortened way to call transpose()
     * @return transposed matrix
     */
    public Matrix T()
    {
        return transpose();
    }

    /**
     * Gets a column of a matrix
     * @param c     the cth column
     * @return      column of matrix
     */
    public double[] getCol(int c)
    {
        double[] ret = new double[this.shape[0]]; // the length of a column is the number of rows

        // for each row, add the member of that column to the array
        for (int i = 0; i < this.shape[0]; i++)
        {
            double[] row = this.matrix[i];
            ret[i]       = row[c];
        }
        return ret;
    }

    /**
     * Prints the matrix
     * object oriented
     */
    public void printMat()
    {
        for (int i = 0; i < this.shape[0]; i++)
        {
            System.out.println(Arrays.toString(this.matrix[i]));
        }
    }

    /**
     * Prints the shape of a matrix
     */
    public void printShape()
    {
        System.out.println("(" + this.shape[0] + ", " + this.shape[1] + ")");
    }

    /**
     * Performs an element-wise multiplecation on matrices of the same size
     * (or if one of the dims is one and the other is
     * the same)
     * aka Hadamard product
     * @return the product matrix
     */
    public Matrix elementMult(Matrix mat2)
    {
        if (this.shape[0] == 1 && this.shape[1] == 1)
        {
            return mat2.scalarMultiply(this.matrix[0][0]);
        }
        if(mat2.shape[0] == 1 && mat2.shape[1] == 1)
        {
            return this.scalarMultiply(mat2.matrix[0][0]);
        }
        boolean sameRows = this.shape[0] == mat2.shape[0];
        boolean sameCols = this.shape[1] == mat2.shape[1];

        boolean canMult = sameRows && sameCols; // can multiply if same dimensions
        if (!canMult)
        {
            // is good if either rows or columns doesn't match, but either (but not both) is 1
            if (!sameRows && sameCols)
            {
                canMult = this.shape[0] == 1 || mat2.shape[0] == 1;
            }
            else if (sameRows && !sameCols)
            {
                canMult = this.shape[1] == 1 || mat2.shape[1] == 1;
            }
        }
        // if you cannot mult them, throw an error
        if (!canMult)
        {
            throw new RuntimeException(
                    "Cannot multiply shapes together: Cannot multiply (" +
                            shape[0] + "x" + shape[1] + ") with ("
                            + mat2.shape[0] + "x" + mat2.shape[1] + ")");
        }

        // make a new matrix, sum, with the dimensions of sum being the max of the dimensions of the matrices being added together
        Matrix product = new Matrix(new double[Math.max(this.shape[0], mat2.shape[0])][Math.max(this.shape[1], mat2.shape[1])]);

        int mat1RowIndex = 0;
        int mat1ColIndex = 0;

        int mat2ColIndex = 0;
        int mat2RowIndex = 0;

        for (int r = 0; r < product.shape[0]; r++)
        {
            mat1ColIndex = 0;
            mat2ColIndex = 0;
            for (int c = 0; c < product.shape[1]; c++)
            {
                product.matrix[r][c] =
                        this.matrix[mat1RowIndex][mat1ColIndex] *
                                mat2.matrix[mat2RowIndex][mat2ColIndex];

                if (mat1ColIndex < this.shape[1] - 1) mat1ColIndex ++;
                if (mat2ColIndex < mat2.shape[1] - 1) mat2ColIndex ++;
            }
            if (mat1RowIndex < this.shape[0] - 1) mat1RowIndex ++;
            if (mat2RowIndex < mat2.shape[0] - 1) mat2RowIndex ++;
        }
        return product;
    }

    /**
     * Performs an element-wise division on matrices of the same size (or if one of the dims is one and the other is
     * the same)
     * @param mat2 the divisor
     * @return the quotient matrix
     */
    public Matrix elementDivide(Matrix mat2)
    {
        if (this.shape[0] == 1 && this.shape[1] == 1)
        {
            return mat2.scalarDivide(this.matrix[0][0]);
        }
        if(mat2.shape[0] == 1 && mat2.shape[1] == 1)
        {
            return this.scalarDivide(mat2.matrix[0][0]);
        }
        boolean sameRows = this.shape[0] == mat2.shape[0];
        boolean sameCols = this.shape[1] == mat2.shape[1];

        boolean canDiv = sameRows && sameCols; // can divide if same dimensions
        if (!canDiv)
        {
            // is good if either rows or columns doesn't match, but either (but not both) is 1
            if (!sameRows && sameCols)
            {
                canDiv = this.shape[0] == 1 || mat2.shape[0] == 1;
            }
            else if (sameRows && !sameCols)
            {
                canDiv = this.shape[1] == 1 || mat2.shape[1] == 1;
            }
        }
        // if you cannot divide them, throw an error
        if (!canDiv)
        {
            throw new RuntimeException(
                    "Cannot divide shapes: Cannot divide (" +
                            shape[0] + "x" + shape[1] + ") with ("
                            + mat2.shape[0] + "x" + mat2.shape[1] + ")");
        }

        // make a new matrix, quotient, with the dimensions of sum being the max of the dimensions of the matrices being added together
        Matrix quotient = new Matrix(new double[Math.max(this.shape[0], mat2.shape[0])][Math.max(this.shape[1], mat2.shape[1])]);

        int mat1RowIndex = 0;
        int mat1ColIndex = 0;

        int mat2ColIndex = 0;
        int mat2RowIndex = 0;

        for (int r = 0; r < quotient.shape[0]; r++)
        {
            mat1ColIndex = 0;
            mat2ColIndex = 0;
            for (int c = 0; c < quotient.shape[1]; c++)
            {
                quotient.matrix[r][c] =
                        this.matrix[mat1RowIndex][mat1ColIndex] /
                                mat2.matrix[mat2RowIndex][mat2ColIndex];

                if (mat1ColIndex < this.shape[1] - 1) mat1ColIndex ++;
                if (mat2ColIndex < mat2.shape[1] - 1) mat2ColIndex ++;
            }
            if (mat1RowIndex < this.shape[0] - 1) mat1RowIndex ++;
            if (mat2RowIndex < mat2.shape[0] - 1) mat2RowIndex ++;

        }
        return quotient;
    }

    /**
     * Multiplies a matrix by a scalar
     * @param s     scalar to multiply matrix by
     * @return      matrix multiplied by scalar
     */
    public Matrix scalarMultiply(double s)
    {
        int x         = this.shape[0];
        int y         = this.shape[1];
        double[][] ret = new double[x][y];

        for (int i = 0; i < x; i++)
        {
            for (int j = 0; j < y; j++)
            {
                ret[i][j] = this.matrix[i][j] * s;
            }
        }
        return new Matrix(ret);
    }

    /**
     * divides each element of a matrix by a scalar
     * @param s scalar to divide everything by
     * @return new matrix of divided values
     */
    public Matrix scalarDivide(double s)
    {
        int x         = this.shape[0];
        int y         = this.shape[1];
        double[][] ret = new double[x][y];

        for (int i = 0; i < x; i++)
        {
            for (int j = 0; j < y; j++)
            {
                ret[i][j] = this.matrix[i][j] / s;
            }
        }
        return new Matrix(ret);
    }

    /**
     * adds a number to each part of matrix
     * @param s     scalar to add to matrix
     * @return      the added matrix
     */
    public Matrix scalarAdd(double s)
    {
        int x         = this.shape[0];
        int y         = this.shape[1];
        double[][] ret = new double[x][y];

        for (int i = 0; i < x; i++)
        {
            for (int j = 0; j < y; j++)
            {
                ret[i][j] = this.matrix[i][j] + s;
            }
        }
        return new Matrix(ret);
    }

    /**
     * Gets the square root of every element in the Matrix
     * @param mat the matrix to get the sqrt of
     * @return a new Matrix of the new values
     */
    public static Matrix sqrt(Matrix mat)
    {
        Matrix returnMat = new Matrix(new double[mat.shape[0]][mat.shape[1]]);
        for (int r = 0; r < mat.shape[0]; r++)
        {
            for (int c = 0; c < mat.shape[1]; c++)
            {
                returnMat.matrix[r][c] = Math.sqrt(mat.matrix[r][c]);
            }
        }
        return returnMat;
    }

    /**
     * Performs sigmoid function on an individual value
     * @param x value
     * @return  transformed "squashed" value
     */
    public static double sigmoidInd(double x)
    {
        return (1 / (1 + Math.exp(-x)));
    }

    /**
     * Performs sigmoid prime function on an individual value
     * @param x value
     * @return  transformed value
     */
    public static double sigmoidPrimeInd(double x)
    {
        return x * (1 - x);
    }

    /**
     * Performs sigmoid function on all values in the matrix and returns a new matrix - it leaves the one you
     * passed in unchanged
     * @return returnMat : matrix of "squashed" values
     */
    public static Matrix sigmoidMat(Matrix mat)
    {
        Matrix returnMat = new Matrix(new double[mat.shape[0]][mat.shape[1]]);
        for (int r = 0; r < mat.shape[0]; r++)
        {
            for (int c = 0; c < mat.shape[1]; c++)
            {
                returnMat.matrix[r][c] = sigmoidInd(mat.matrix[r][c]);
            }
        }
        return returnMat;
    }

    /**
     * Performs the sigmoid prime function on all values in the matrix and returns a new matrix - it leaves the one you
     * passed in unchanged
     * @return returnMat
     */
    public static Matrix sigmoidPrimeMat(Matrix mat)
    {
        Matrix returnMat = new Matrix(new double[mat.shape[0]][mat.shape[1]]);
        for (int r = 0; r < mat.shape[0]; r++)
        {
            for (int c = 0; c < mat.shape[1]; c++)
            {
                returnMat.matrix[r][c] = sigmoidPrimeInd(mat.matrix[r][c]);
            }
        }
        return returnMat;
    }

    /**
     * Performs tanh (hyperbolic tangent) function on an individual value
     *
     * There are lots of ways to write it out, this one seemed to be the most computationally efficient.
     * Here are some of the others along with the graph: https://www.desmos.com/calculator/mwrb27cgpg
     * @param x value
     * @return  transformed value
     */
    public static double tanhInd(double x)
    {
        return 2.0/(1.0 + Math.exp(-2 * x)) - 1;
    }

    /**
     * Performs the derivative of tanh on an individual value
     * 1 - f(x)^2
     * @param x
     * @return
     */
    public static double tanhPrimeInd(double x)
    {
        return 1 - Math.pow(tanhInd(x), 2.0);
    }

    /**
     * Performs tanh function on all values in the matrix and returns a new matrix - it leaves the one you
     * passed in unchanged
     * @return returnMat : matrix of "squashed" values
     */
    public static Matrix tanhMat(Matrix mat)
    {
        Matrix returnMat = new Matrix(new double[mat.shape[0]][mat.shape[1]]);
        for (int r = 0; r < mat.shape[0]; r++)
        {
            for (int c = 0; c < mat.shape[1]; c++)
            {
                returnMat.matrix[r][c] = tanhInd(mat.matrix[r][c]);
            }
        }
        return returnMat;
    }

    /**
     * Performs derivative of the tanh function on all values in the matrix and returns a new matrix - it leaves the one you
     * passed in unchanged
     * @return returnMat : matrix of values
     */
    public static Matrix tanhPrimeMat(Matrix mat)
    {
        Matrix returnMat = new Matrix(new double[mat.shape[0]][mat.shape[1]]);
        for (int r = 0; r < mat.shape[0]; r++)
        {
            for (int c = 0; c < mat.shape[1]; c++)
            {
                returnMat.matrix[r][c] = tanhPrimeInd(mat.matrix[r][c]);
            }
        }
        return returnMat;
    }

    /**
     * does e^x for each element in the matrix
     * @param mat
     * @return the new matrix of changed elements
     */
    public static Matrix expMat(Matrix mat)
    {
        Matrix returnMat = new Matrix(new double[mat.shape[0]][mat.shape[1]]);
        for (int r = 0; r < mat.shape[0]; r++)
        {
            for (int c = 0; c < mat.shape[1]; c++)
            {
                returnMat.matrix[r][c] = Math.exp(mat.matrix[r][c]);
            }
        }
        return returnMat;
    }

    /**
     * takes all the elements in a matrix and adds them up
     * @return the sum
     */
    public static double sum(Matrix mat)
    {
        double sum = 0;
        for (int r = 0; r < mat.shape[0]; r++)
        {
            for (int c = 0; c < mat.shape[1]; c++)
            {
                sum += mat.matrix[r][c];
            }
        }
        return sum;
    }

    /**
     * Softmax activation function
     * https://en.wikipedia.org/wiki/Activation_function
     * and https://en.wikipedia.org/wiki/Softmax_function
     * "to highlight the largest values and suppress values which are significantly below the maximum value"
     *
     * @param mat the matrix to perform the softmax function on
     * @return the new matrix with new values
     */
    public static Matrix softmax(Matrix mat)
    {
        Matrix numerator = expMat(mat);
        double denominator = sum(numerator);
        return numerator.scalarDivide(denominator);
    }

    /**
     * Save a matrix to a certain file
     * @param path the file to save the array to
     */
    public void save(String path) throws java.io.IOException
    {
        File toSaveFile = new File(path);
        // make the file if it doesn't exist
        if (!toSaveFile.exists())
        {
            toSaveFile.createNewFile();
        }

        FileWriter fw = new FileWriter(toSaveFile, true);

        // each matrix is on its own line with the rows separated by '*'
        StringBuilder toWrite = new StringBuilder();
        for (int r = 0; r < this.shape[0]; r++)
        {
            String stringMat = Arrays.toString(this.matrix[r]).replace("[", "");
            stringMat = stringMat.replace("]", "");
            stringMat = stringMat.replace(" ", "");
            stringMat = stringMat.replace("0.0", "&"); // there are going to be a lot of 0.0, so it makes sense to shorten it
            toWrite.append(stringMat + "*");
        }
        String stringArray = toWrite + "";
        stringArray = stringArray.replace("&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*", "^"); // replace 20 zeros with ^
        fw.write(stringArray);
        fw.write("\n");
        fw.close();
    }

    /**
     * Get matrices from things saved by save()
     * @param path the path to the text file where matrices were saved
     * @return an array of matrices
     * @throws java.io.IOException
     */
    public static Matrix[] load(String path) throws java.io.IOException
    {
        // get the contents of the file
        File savedFile = new File(path);
        Scanner sc = new Scanner(savedFile);

        ArrayList<String> splitUp = new ArrayList<>();
        while (sc.hasNextLine())
        {
            splitUp.add(sc.nextLine());
        }

        Matrix[] allMatrices = new Matrix[splitUp.size()];

        // for each matrix
        for (int s = 0; s < splitUp.size(); s++)
        {
            String string = splitUp.get(s);
            string = string.replace("^", "&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*");

            String[] byRow = string.split("\\*");

            double[][] matrix = new double[byRow.length][];

            // for each row in matrix
            for (int r = 0; r < byRow.length; r++)
            {
                String row = byRow[r];
                row = row.replace("&", "0.0"); // this was a way to save memory space when saving
                String[] numsInRow = row.split(",");
                double[] matRow = new double[numsInRow.length];

                // for each num in row add it to the row array
                for (int c = 0; c < numsInRow.length; c++)
                {
                    matRow[c] = Double.parseDouble(numsInRow[c]);;
                }
                matrix[r] = matRow;
            }
            allMatrices[s] = new Matrix(matrix);
        }
        return allMatrices;
    }

    /**
     * returns the row and column of the maximum number
     * @return
     */
    public int[] getMaxIndex()
    {
        int[] maxIndex = new int[] {-1, -1};
        double max = Double.MIN_VALUE;

        for (int r = 0; r < this.shape[0]; r++)
        {
            for (int c = 0; c < this.shape[1]; c++)
            {
                if (this.matrix[r][c] > max)
                {
                    max = this.matrix[r][c];
                    maxIndex = new int[] {r, c};
                }
            }
        }

        return maxIndex;
    }

    /**
     * Fills a 2d matrix array with a certain matrix
     * @param arrayDims the dimensions of the matrix array
     * @param indivMat the double[][] that each spot will be filled with
     */
    public static Matrix[][] fill2dMatArray(int[] arrayDims, double[][] indivMat)
    {
        Matrix[][] returnArray = new Matrix[arrayDims[0]][arrayDims[1]];
        for (int d1 = 0; d1 < arrayDims[0]; d1++)
        {
            for (int d2 = 0; d2 < arrayDims[1]; d2++)
            {
                returnArray[d1][d2] = new Matrix(indivMat);
            }
        }
        return returnArray;
    }
}